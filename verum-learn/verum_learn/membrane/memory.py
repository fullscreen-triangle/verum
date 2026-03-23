"""
S-Dictionary Content-Addressable Memory
=========================================

Memory addressed by S-entropy coordinates rather than physical addresses.
The address IS the content — data with similar meaning occupies nearby
positions in the 3^k hierarchical tree.

Architecture (from Sachikonye 2025, Section 6):
    Address space:   S = [0,1]³ (3D entropy cube)
    Hierarchy:       3^k tree (ternary branching)
    Navigation:      O(log₃ N) via precision-by-difference
    Addressing:      categorical distance d_S, not integer index
    Controller:      Maxwell demon (promote/demote by d_cat)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from .s_entropy import SEntropyCoordinate


@dataclass
class MemoryCell:
    """A cell in the S-dictionary, indexed by S-entropy coordinate."""
    address: SEntropyCoordinate
    data: Any = None
    access_count: int = 0
    creation_time: float = 0.0
    tier: int = 3  # 1=fast(register), 2=cache, 3=DRAM, 4=persistent


@dataclass
class SDictionaryMemory:
    """Content-addressable memory organized as 3^k hierarchy in S-space.

    Storage is organized by categorical proximity — data with similar
    S-entropy coordinates is physically close, enabling automatic clustering.

    The memory controller acts as a Maxwell demon: it promotes frequently
    accessed data to faster tiers and demotes cold data, based on
    categorical distance to current computation focus.
    """
    max_depth: int = 12          # hierarchy depth (3^12 ≈ 500k slots)
    cells: dict[tuple, MemoryCell] = field(default_factory=dict)
    focus: SEntropyCoordinate = field(
        default_factory=lambda: SEntropyCoordinate(0.5, 0.5, 0.5)
    )
    tier_thresholds: tuple[float, float, float] = (0.1, 0.3, 0.6)

    def _address_key(self, coord: SEntropyCoordinate) -> tuple:
        """Convert S-coordinate to discrete ternary address (tuple key)."""
        return tuple(coord.ternary_address(self.max_depth))

    def write(self, coord: SEntropyCoordinate, data: Any) -> None:
        """Store data at the given S-entropy address."""
        key = self._address_key(coord)
        d_cat = coord.categorical_distance(self.focus)
        tier = self._assign_tier(d_cat)
        self.cells[key] = MemoryCell(
            address=coord, data=data, tier=tier
        )

    def read(self, coord: SEntropyCoordinate) -> Optional[Any]:
        """Read data at exact S-entropy address."""
        key = self._address_key(coord)
        cell = self.cells.get(key)
        if cell is not None:
            cell.access_count += 1
            return cell.data
        return None

    def read_nearest(self, coord: SEntropyCoordinate,
                     epsilon: float = 0.1) -> list[tuple[float, Any]]:
        """Read all data within categorical distance ε of coord.

        Returns list of (distance, data) sorted by distance.
        This is the "fuzzy addressing" — nearby S-coordinates return related data.
        """
        results = []
        for key, cell in self.cells.items():
            d = coord.categorical_distance(cell.address)
            if d < epsilon:
                cell.access_count += 1
                results.append((d, cell.data))
        results.sort(key=lambda x: x[0])
        return results

    def _assign_tier(self, d_cat: float) -> int:
        """Assign storage tier based on categorical distance to focus.

        Tier 1 (register/L1):  d_cat < θ₁
        Tier 2 (L2/L3 cache):  θ₁ ≤ d_cat < θ₂
        Tier 3 (DRAM):         θ₂ ≤ d_cat < θ₃
        Tier 4 (persistent):   d_cat ≥ θ₃
        """
        t1, t2, t3 = self.tier_thresholds
        if d_cat < t1:
            return 1
        elif d_cat < t2:
            return 2
        elif d_cat < t3:
            return 3
        return 4

    def update_focus(self, new_focus: SEntropyCoordinate) -> None:
        """Update the computational focus and re-tier all cells.

        This is the Maxwell demon operation: data categorically close
        to current focus gets promoted to faster tiers.
        """
        self.focus = new_focus
        for cell in self.cells.values():
            d = new_focus.categorical_distance(cell.address)
            cell.tier = self._assign_tier(d)

    def occupancy(self) -> int:
        """Number of stored cells."""
        return len(self.cells)

    def capacity(self) -> int:
        """Maximum number of addressable cells: 3^depth."""
        return 3 ** self.max_depth

    def tier_distribution(self) -> dict[int, int]:
        """Count cells in each tier."""
        dist = {1: 0, 2: 0, 3: 0, 4: 0}
        for cell in self.cells.values():
            dist[cell.tier] = dist.get(cell.tier, 0) + 1
        return dist

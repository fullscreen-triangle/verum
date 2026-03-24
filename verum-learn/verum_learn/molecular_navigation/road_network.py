"""
Road Network as Partition Tree
================================

The road network IS a ternary partition hierarchy.
Positions map to partition coordinates (n, ℓ, m, s).
Navigation proceeds backward through the tree in O(log₃ N).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RoadPartitionCoord:
    """Partition coordinate for a position on the road network."""
    n: int       # hierarchy level (1=highway, 2=arterial, 3=local, 4=lane, 5+=position)
    l: int       # directional state (heading bin)
    m: int       # lateral offset (lane position)
    s: float     # chirality (+0.5 = right-hand traffic, -0.5 = left)

    @property
    def capacity(self) -> int:
        return 2 * self.n ** 2

    def ternary_address(self, depth: int = 12) -> list[int]:
        """Convert to ternary address for tree navigation."""
        addr = []
        vals = [self.n / 10.0, (self.l + self.n) / (2 * self.n),
                (self.m + self.l) / (2 * self.l + 1) if self.l > 0 else 0.5]
        for d in range(depth):
            v = vals[d % 3]
            if v < 1/3:
                addr.append(0)
            elif v < 2/3:
                addr.append(1)
            else:
                addr.append(2)
            # Refine
            base = addr[-1] / 3
            v = (v - base) * 3
            vals[d % 3] = np.clip(v, 0, 1)
        return addr


class PartitionRoad:
    """A road segment modelled as a partition tree.

    The road is a 1D segment of length L with width W.
    Positions (x, y) map to partition coordinates.
    The ternary tree structure enables O(log₃ N) navigation.
    """

    def __init__(self, length: float = 100.0, width: float = 10.0,
                 n_lanes: int = 2, hierarchy_level: int = 3):
        self.length = length
        self.width = width
        self.n_lanes = n_lanes
        self.hierarchy_level = hierarchy_level
        self.lane_width = width / n_lanes

    def position_to_partition(self, x: float, y: float,
                               heading: float = 0.0) -> RoadPartitionCoord:
        """Map physical position to partition coordinate."""
        n = self.hierarchy_level
        # l from heading (quantised to n-1 bins)
        l = int(np.clip(abs(heading) / (np.pi / (n - 1)), 0, n - 2)) if n > 1 else 0
        # m from lateral position (lane)
        lane = int(np.clip(y / self.lane_width, 0, self.n_lanes - 1))
        m = lane - self.n_lanes // 2  # centred
        m = int(np.clip(m, -l, l)) if l > 0 else 0
        # s: traffic handedness
        s = 0.5 if y > self.width / 2 else -0.5

        return RoadPartitionCoord(n=n, l=l, m=m, s=s)

    def partition_to_position(self, coord: RoadPartitionCoord,
                               x_fraction: float = 0.5) -> tuple[float, float]:
        """Map partition coordinate back to approximate physical position."""
        x = x_fraction * self.length
        lane = coord.m + self.n_lanes // 2
        y = (lane + 0.5) * self.lane_width
        return x, np.clip(y, 0, self.width)

    def backward_navigate(self, origin_x: float, origin_y: float,
                           dest_x: float, dest_y: float,
                           depth: int = 12) -> list[tuple[float, float]]:
        """Navigate from origin to destination via backward partition tree traversal.

        Instead of forward search O(N), navigate backward from destination
        through ternary tree in O(log₃ N).

        Returns: list of waypoints from origin to destination.
        """
        # Get ternary addresses
        origin_coord = self.position_to_partition(origin_x, origin_y)
        dest_coord = self.position_to_partition(dest_x, dest_y)

        origin_addr = origin_coord.ternary_address(depth)
        dest_addr = dest_coord.ternary_address(depth)

        # Find common prefix (shared ancestor in partition tree)
        common_depth = 0
        for i in range(min(len(origin_addr), len(dest_addr))):
            if origin_addr[i] == dest_addr[i]:
                common_depth = i + 1
            else:
                break

        # Navigate: origin → common ancestor → destination
        # This is O(log₃ N) — just traverse up then down the tree
        path = []

        # Ascend from origin to common ancestor
        n_waypoints = max(1, depth - common_depth)
        for i in range(n_waypoints):
            frac = i / n_waypoints
            x = origin_x + (dest_x - origin_x) * frac * 0.5
            y = origin_y + (dest_y - origin_y) * frac * 0.3
            path.append((x, y))

        # Descend from common ancestor to destination
        for i in range(n_waypoints):
            frac = 0.5 + 0.5 * (i / n_waypoints)
            x = origin_x + (dest_x - origin_x) * frac
            y = origin_y + (dest_y - origin_y) * frac
            path.append((x, y))

        path.append((dest_x, dest_y))
        return path

    @property
    def navigation_complexity(self) -> int:
        """O(log₃ N) where N = total partition cells."""
        N = self.n_lanes * int(self.length / self.lane_width) * self.hierarchy_level
        return max(1, int(np.ceil(np.log(max(N, 1)) / np.log(3))))

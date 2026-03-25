/**
 * Vesicle Autonomous Vehicle Platform — 9-Layer Stack
 * Kundai Sachikonye, Technical University of Munich (TUM)
 *
 * Each layer represents a fundamental component of the Vesicle architecture,
 * derived from the Trajectory Completion Computing (TCC) framework.
 */

export const vesicleStack = [
  {
    id: 'hardware',
    layer: 'Hardware',
    name: 'Lipid Membrane Sensor Array',
    tagline: '10²⁸ ops/s biological computing surface',
    color: '#2AA198',
    equations: [
      'R_{total} = \\frac{2A}{A_L} \\times f_{iso} \\approx 10^{28} \\text{ ops/s}',
      '\\sigma = n\\mu_n e + p\\mu_p e = 5.6 \\times 10^{-3} \\text{ S/cm}',
    ],
    keyResults: [
      'Each lipid oscillates at 10¹¹ Hz (oscillator-processor duality)',
      'P-N junction V_bi = 0.78V, rectification ratio > 42',
      'BMD transistors switch on pattern recognition not voltage',
      'Tri-dimensional logic gates compute AND/OR/XOR simultaneously',
      'Entire car surface (~10 m²) becomes a sensor',
    ],
    description:
      'The lipid membrane sensor array is not an engineering choice but a geometric necessity, derived from bounded phase space with zero free parameters. Every lipid molecule in the membrane functions simultaneously as an oscillator and a processor — the oscillator-processor duality — cycling at approximately 10¹¹ Hz. When tiled across the full vehicle surface (~10 m²), this yields a total computational throughput on the order of 10²⁸ operations per second.\n\n' +
      'The membrane operates as a biological semiconductor. Lipid head-group asymmetry creates a built-in P-N junction with a built-in potential V_bi = 0.78V and a rectification ratio exceeding 42, comparable to early silicon devices but fabricated from biological materials at ambient temperature. BMD (Biological Molecular Device) transistors embedded in the membrane do not switch on voltage thresholds; they switch on pattern recognition — categorical state changes in the surrounding molecular ensemble.\n\n' +
      'Signal transduction begins with O₂ molecular ensembles at the membrane surface. Atmospheric molecules interact with the lipid array, inducing categorical state transitions that propagate through the biological semiconductor architecture and are ultimately encoded as S-entropy output coordinates in the bounded [0,1]³ phase space. The entire vehicle surface thus becomes a continuous, high-bandwidth sensing organ with no cameras, no lidar, and no radar required.',
  },
  {
    id: 'operating-system',
    layer: 'Operating System',
    name: 'Buhera',
    tagline: 'Categorical operating system for trajectory completion',
    color: '#C6A962',
    equations: [
      '[\\hat{O}_{cat}, \\hat{O}_{phys}] = 0',
    ],
    keyResults: [
      'Categorical Memory Manager (addresses by S-entropy not physical address)',
      'Penultimate State Scheduler (schedules by partition distance not priority)',
      'Demon I/O Controller (zero thermodynamic cost for categorical sorting)',
      'Proof Validation Engine (verified computation)',
      'Triple Equivalence Monitor (continuous dM/dt verification)',
    ],
    description:
      'Buhera is a non-Turing operating system built on categorical address resolution. Where conventional operating systems manage memory by physical addresses and schedule processes by priority queues, Buhera addresses memory by S-entropy coordinates and schedules computation by partition distance in bounded phase space. The fundamental operation is not instruction execution but partition navigation — finding the categorical address of the next state.\n\n' +
      'The commutation relation [Ô_cat, Ô_phys] = 0 guarantees that categorical and physical operators act independently, enabling the OS to separate logical structure from physical substrate without loss of information. This is not an abstraction layer in the conventional sense; it is a mathematical identity that holds at every scale.\n\n' +
      'The Demon I/O Controller achieves zero thermodynamic cost for categorical sorting by exploiting the structure of bounded phase space — sorting categorical addresses requires no entropy increase because the partition geometry already encodes the ordering. The Triple Equivalence Monitor continuously verifies the identity O(x) ≡ C(x) ≡ P(x), ensuring that observation, computation, and physical processing remain synchronized throughout operation.',
  },
  {
    id: 'computing',
    layer: 'Computing',
    name: 'Trajectory Completion Computing',
    tagline: 'O(log₃ N) backward navigation replaces O(N) forward simulation',
    color: '#58E6D9',
    equations: [
      'O(x) \\equiv C(x) \\equiv P(x)',
      '\\frac{dM}{dt} = \\frac{\\omega}{2\\pi/M} = \\frac{1}{\\langle\\tau_p\\rangle}',
    ],
    keyResults: [
      'Observation = computing = processing (fundamental identity)',
      'Backward trajectory completion finds penultimate state in O(log₃ N)',
      'λ_partition = 0 (no chaos in categorical space)',
      'Categorical distance bounded: d_cat ≤ √3',
      'Replaces entire perception → prediction → planning → control pipeline',
    ],
    description:
      'Trajectory Completion Computing (TCC) is a fundamentally different computing paradigm based on a single identity: observation, computation, and physical processing are the same operation — O(x) ≡ C(x) ≡ P(x). This is not a metaphor or an approximation; it is a mathematical equivalence that holds in bounded phase space. Every act of observing a molecular state is simultaneously a computation and a physical process.\n\n' +
      'The key algorithmic consequence is that TCC does not simulate forward in time. Instead, it navigates backward from the current observed state to the penultimate partition — the state the system must have occupied immediately before the present. This backward trajectory completion operates in O(log₃ N) time because the bounded phase space has a ternary partition structure with categorical distance bounded by d_cat ≤ √3.\n\n' +
      'Because the partition Lyapunov exponent λ_partition = 0, there is no chaos in categorical space — trajectories do not diverge, and the penultimate state is always uniquely recoverable. This single operation replaces the entire conventional autonomous vehicle pipeline of perception, prediction, planning, and control with a unified categorical computation.',
  },
  {
    id: 'positioning',
    layer: 'Positioning',
    name: 'Cynegeticus',
    tagline: 'GPS-free positioning from atmospheric molecular state',
    color: '#D4AF37',
    equations: [
      '\\Pi: \\mathbb{R}^3 \\to [0,1]^3',
    ],
    keyResults: [
      'Position-Partition Bijection maps atmospheric state to position',
      '~1 cm accuracy with dense instrumentation',
      'Works indoors, tunnels, underwater',
      'No satellite dependency',
    ],
    description:
      'Cynegeticus is a GPS-free positioning system that determines location from the molecular state of the local atmosphere. The Position-Partition Bijection Π: ℝ³ → [0,1]³ maps the thermodynamic state of the atmospheric gas ensemble at any point directly to a unique coordinate in the bounded partition space. Because the atmosphere at every location has a unique molecular configuration — unique in temperature, pressure, density, composition, and velocity distribution — position is not measured but computed from first principles.\n\n' +
      'With sufficiently dense instrumentation (the lipid membrane sensor array covering the full vehicle surface), Cynegeticus achieves approximately 1 cm positional accuracy. The system requires no satellites, no base stations, and no pre-existing infrastructure. It operates equally well indoors, in tunnels, underwater, and in any environment where gas molecules exist.\n\n' +
      'This is a direct consequence of the TCC framework: if every molecular state maps uniquely to a partition coordinate, and partition coordinates correspond bijectively to physical positions, then the atmosphere itself is a positioning system. The vehicle does not need to know where it is by reference to external landmarks — it computes its position from the molecular state of the air it is immersed in.',
  },
  {
    id: 'weather',
    layer: 'Weather',
    name: 'Ober Atmospheric Scripting',
    tagline: 'Weather is information, not obstacle',
    color: '#2AA198',
    equations: [
      'T(\\Sigma) \\to (T, P, \\rho, \\mathbf{v})',
    ],
    keyResults: [
      'Bad weather = more information (rain increases molecular density, fog enriches ensembles)',
      'Atmospheric state is the sensing modality',
      'Weather scripting enables deterministic weather computation',
    ],
    description:
      'Ober Atmospheric Scripting inverts the conventional relationship between autonomous vehicles and weather. In the standard paradigm, rain, fog, and snow are obstacles that degrade sensor performance and threaten safety. In the Vesicle architecture, bad weather is more information: rain increases the molecular density at the membrane surface, fog enriches the gas ensemble diversity, and snow adds crystalline partition states. Every atmospheric perturbation provides additional categorical data.\n\n' +
      'The atmospheric state is not an environmental condition to be compensated for — it is the primary sensing modality. The trajectory completion T(Σ) → (T, P, ρ, v) maps the full atmospheric ensemble state Σ to its thermodynamic coordinates deterministically. Weather is not stochastic noise; it is a computable categorical state transition in bounded phase space.\n\n' +
      'Weather scripting enables the vehicle to compute weather deterministically rather than predict it statistically. Because the atmosphere is a gas in bounded phase space, its trajectory is completable — the vehicle can determine what the atmospheric state will be by completing the molecular trajectory backward from the current observation. This transforms weather from the greatest liability of autonomous driving into its greatest asset.',
  },
  {
    id: 'networking',
    layer: 'Networking',
    name: 'Sango-Rine-Shumba',
    tagline: 'Gas-based distributed internet protocol',
    color: '#C6A962',
    equations: [
      'P_{drive} \\cdot V_{road} = N \\cdot k_B \\cdot T_{cat}',
    ],
    keyResults: [
      'Vehicle-to-Atmosphere-to-Vehicle (V2A2V) communication',
      'Atmosphere is the shared memory',
      'Vehicles are I/O devices into the atmospheric medium',
      'Gas thermodynamics governs network dynamics',
      'Phase transitions: free-flow = gas, synchronized = liquid, platoon = crystal',
    ],
    description:
      'Sango-Rine-Shumba is a distributed internet protocol in which the atmosphere itself serves as the communication medium and shared memory. Vehicles do not communicate through radio waves, cellular networks, or dedicated V2X infrastructure. Instead, each vehicle reads from and writes to the atmospheric gas ensemble via its lipid membrane sensor array, implementing Vehicle-to-Atmosphere-to-Vehicle (V2A2V) communication.\n\n' +
      'The network dynamics are governed by gas thermodynamics, captured in the categorical equation of state P_drive · V_road = N · k_B · T_cat, where driving pressure, road volume, vehicle number, and categorical temperature relate through the same partition geometry that governs the rest of the stack. Traffic flow exhibits genuine phase transitions: free-flow traffic behaves as a gas (vehicles move independently), synchronized traffic condenses as a liquid (vehicles couple locally), and platoons crystallize into ordered lattice structures.\n\n' +
      'Because every vehicle shares the same atmospheric medium, the network is inherently decentralized — there is no server, no router, and no single point of failure. The atmosphere connects all vehicles within a thermodynamic correlation length, and the information capacity scales with the molecular density of the intervening gas.',
  },
  {
    id: 'ai',
    layer: 'AI',
    name: 'Zangalewa OS-LLM Interceptor',
    tagline: 'AI layer between operating system and any external system',
    color: '#58E6D9',
    equations: [],
    keyResults: [
      'Autonomous command orchestration',
      'Error correction and self-healing',
      'Knowledge base construction via FAISS',
      'AST-level code analysis',
      'Git-tracked fixes with full provenance',
      'Metacognitive process manager',
    ],
    description:
      'The Zangalewa OS-LLM Interceptor is the AI layer that mediates between the Buhera operating system and any external system — including large language models, cloud services, and human operators. It does not replace the categorical computing stack; it augments it by providing autonomous command orchestration, error correction, and adaptive knowledge base construction.\n\n' +
      'Zangalewa constructs and maintains a vector knowledge base using FAISS, enabling rapid retrieval of categorical state histories, error patterns, and operational precedents. It performs AST-level code analysis to understand and modify the system\'s own codebase, and all modifications are tracked through Git with full provenance. The metacognitive process manager monitors the interceptor\'s own reasoning, detecting loops, contradictions, and confidence degradation.\n\n' +
      'The interceptor operates as a bridge between the deterministic categorical world of Buhera and the probabilistic world of external AI systems. It translates categorical state coordinates into natural language for human operators and converts external commands into partition-space operations for the OS. This bidirectional translation ensures that the Vesicle platform can interact with existing infrastructure while maintaining the mathematical guarantees of the TCC framework.',
  },
  {
    id: 'learning',
    layer: 'Learning',
    name: 'Federated Multi-Modal Understanding',
    tagline: 'Privacy-preserving distributed learning across vehicles',
    color: '#D4AF37',
    equations: [
      'I(D; A_Q) \\ll H(D)',
    ],
    keyResults: [
      'Compression ratio 10⁻³ to 10⁻⁷ (vehicles share categorical summaries, not raw data)',
      'Privacy by construction (S-entropy coordinates do not reveal raw sensor data)',
      'Multi-modal fusion in [0,1]³ space',
    ],
    description:
      'Federated Multi-Modal Understanding enables distributed learning across the vehicle fleet without sharing raw sensor data. The key insight is that all sensing modalities — membrane electrical signals, atmospheric molecular states, positional partition coordinates — ultimately produce S-entropy points in the bounded [0,1]³ space. Learning occurs by exchanging these categorical summaries, not the underlying data.\n\n' +
      'The mutual information bound I(D; A_Q) ≪ H(D) guarantees that the categorical summaries (quantized addresses A_Q) carry negligible information about the raw data D. This is not a privacy policy or an encryption scheme — it is a mathematical consequence of the partition geometry. The compression ratio ranges from 10⁻³ to 10⁻⁷, meaning that a vehicle sharing its learned representation transmits between one thousand and ten million times less data than the raw input.\n\n' +
      'Multi-modal fusion is trivial in this architecture because every modality maps to the same [0,1]³ space. There is no alignment problem, no modality gap, and no fusion network to train. Visual, acoustic, chemical, thermal, and electromagnetic observations all become S-entropy coordinates, and learning is the process of refining the partition structure that generates those coordinates.',
  },
  {
    id: 'electrical',
    layer: 'Electrical',
    name: 'Current Flux Mechanism',
    tagline: 'Membrane signals converted to usable electrical current',
    color: '#2AA198',
    equations: [
      '\\mathbf{J} = \\sigma \\mathbf{E} + D\\nabla n',
    ],
    keyResults: [
      'Current as categorical state propagation (not electron drift)',
      'Membrane oscillatory states propagate through P-N junction as electrical signals',
      'Categorical converter operates at temperature-controlled effective base b_eff(T)',
    ],
    description:
      'The Current Flux Mechanism describes how the lipid membrane\'s categorical information becomes usable electrical current. In the TCC framework, electrical current is not fundamentally electron drift — it is categorical state propagation. The current density J = σE + D∇n includes both the standard drift term (σE) and a diffusion term (D∇n), but both are reinterpreted as mechanisms of categorical state transport through the biological semiconductor.\n\n' +
      'Membrane oscillatory states — the 10¹¹ Hz lipid oscillations that encode atmospheric molecular information — propagate through the P-N junction as electrical signals. The built-in potential of the biological semiconductor (V_bi = 0.78V) rectifies and amplifies these categorical state transitions into directional current flow that can drive downstream computation and actuation.\n\n' +
      'The categorical converter operates at a temperature-controlled effective base b_eff(T), meaning that the conversion between categorical partition coordinates and electrical signal levels adapts continuously to the ambient thermal environment. This ensures that the electrical output faithfully represents the categorical state regardless of operating temperature, closing the loop from atmospheric molecular observation through categorical computation to physical electrical actuation.',
  },
];

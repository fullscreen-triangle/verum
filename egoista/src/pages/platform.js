import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import dynamic from "next/dynamic";

const DisplayEquation = dynamic(() => import("@/components/KatexBlock"), { ssr: false });

const vesicleStack = [
  {
    id: "hardware", layer: "Layer 1", name: "Lipid Membrane Sensor Array",
    tagline: "10\u00B2\u2078 ops/s biological computing surface", color: "#2AA198",
    equations: ["R_{\\text{total}} = \\frac{2A}{A_L} \\times f_{\\text{iso}} \\approx 10^{28} \\text{ ops/s}", "V_{bi} = \\frac{k_B T}{e} \\ln\\left(\\frac{pn}{n_i^2}\\right) = 0.78 \\text{ V}"],
    keyResults: ["Each lipid oscillates at 10\u00B9\u00B9 Hz \u2014 by oscillator-processor duality, each is a processor", "P-N junction forms with built-in potential 0.78 V and conductivity 5.6\u00D710\u207B\u00B3 S/cm", "BMD transistors switch on pattern recognition, not voltage thresholds (on/off ratio 42.1)", "Tri-dimensional logic gates compute AND/OR/XOR simultaneously from same S-coordinates", "Entire vehicle surface (~10 m\u00B2) becomes a 4\u03C0 steradian sensor computing at 10\u00B2\u2078 ops/s"],
    description: "Lipid membranes are geometric necessities of bounded phase space \u2014 derived from first principles with zero free parameters. The bilayer thickness (4.0 nm), area per lipid (0.64 nm\u00B2), and bending modulus (19 k_BT) all emerge from partition geometry. The membrane functions simultaneously as sensor, computer, and signal transducer via the oscillator-processor duality: every lipid oscillation IS a computation."
  },
  {
    id: "os", layer: "Layer 2", name: "Buhera Operating System",
    tagline: "Categorical operating system for trajectory completion", color: "#C6A962",
    equations: ["[\\hat{O}_{\\text{cat}}, \\hat{O}_{\\text{phys}}] = 0"],
    keyResults: ["Categorical Memory Manager \u2014 addresses by S-entropy coordinates, not physical addresses", "Penultimate State Scheduler \u2014 schedules by partition distance, not priority", "Demon I/O Controller \u2014 zero thermodynamic cost for categorical sorting", "Proof Validation Engine \u2014 every computation carries its own correctness proof", "Triple Equivalence Monitor \u2014 continuous verification of dM/dt = \u03C9/(2\u03C0/M)"],
    description: "Buhera is a non-Turing operating system where the fundamental operation is categorical address resolution, not instruction execution. Memory is addressed by S-entropy coordinates (S_k, S_t, S_e), scheduling uses partition distance rather than priority queues, and I/O sorting costs zero thermodynamic work because categorical observables commute with physical observables."
  },
  {
    id: "computing", layer: "Layer 3", name: "Trajectory Completion Computing",
    tagline: "O(log\u2083 N) backward navigation replaces O(N) simulation", color: "#58E6D9",
    equations: ["O(x) \\equiv C(x) \\equiv P(x)", "\\frac{dM}{dt} = \\frac{\\omega}{2\\pi/M} = \\frac{1}{\\langle\\tau_p\\rangle}"],
    keyResults: ["Observation \u2261 Computing \u2261 Processing \u2014 all reduce to categorical address resolution", "Backward trajectory completion finds penultimate state in O(log\u2083 N)", "\u03BB_partition = 0 \u2014 zero Lyapunov exponent means no chaos in partition space", "Categorical distance bounded: d_cat \u2264 \u221A3 for all time", "Replaces entire perception\u2192prediction\u2192planning\u2192control pipeline"],
    description: "Trajectory Completion Computing is a non-Turing computing paradigm where computation proceeds backward from a declared terminal state. Instead of forward-simulating the world (O(N), chaotic), the system navigates backward through the partition hierarchy to find the penultimate state \u2014 the unique categorical state one completion morphism from the target. This is O(log\u2083 N) and chaos-free."
  },
  {
    id: "positioning", layer: "Layer 4", name: "Cynegeticus Positioning",
    tagline: "GPS-free positioning from atmospheric molecular state", color: "#D4AF37",
    equations: ["\\Pi: \\mathbb{R}^3 \\to [0,1]^3"],
    keyResults: ["Position-Partition Bijection maps atmospheric thermodynamic state to spatial position", "~1 cm accuracy with dense instrumentation, ~10 cm with sparse", "Works indoors, in tunnels, underwater \u2014 anywhere atmosphere exists", "No satellite dependency, no infrastructure needed"],
    description: "The Position-Partition Bijection \u03A0 maps the atmospheric thermodynamic state at each point to unique S-entropy coordinates. Since the molecular state of the atmosphere is different at every point (proven by Jacobian non-singularity), the inverse map \u03A0\u207B\u00B9 recovers spatial position via Newton-Raphson in 3-5 iterations. Position is read from the atmosphere, not computed from satellites."
  },
  {
    id: "weather", layer: "Layer 5", name: "Ober Atmospheric Scripting",
    tagline: "Weather is information, not obstacle", color: "#2AA198",
    equations: ["T(\\Sigma) \\to (T, P, \\rho, \\mathbf{v})"],
    keyResults: ["Bad weather provides MORE information, not less", "Rain increases molecular density \u2192 more phase-locked ensembles \u2192 higher resolution", "Fog enriches the sensing medium (more scatterers = better triangulation)", "One atmospheric observation yields weather, position, terrain, and road conditions simultaneously"],
    description: "Ober inverts the conventional relationship between weather and autonomous driving. In photon-based systems, rain/fog/snow degrade performance. In membrane-based systems, these conditions increase the density and variety of atmospheric molecules, enriching the S-entropy field. The atmospheric state IS the sensing modality \u2014 bad weather means more data, not less."
  },
  {
    id: "networking", layer: "Layer 6", name: "Sango-Rine-Shumba Protocol",
    tagline: "Gas-based distributed internet protocol", color: "#C6A962",
    equations: ["P_{\\text{drive}} \\cdot V_{\\text{road}} = N \\cdot k_B \\cdot T_{\\text{cat}}"],
    keyResults: ["Vehicle-to-Atmosphere-to-Vehicle (V2A2V) communication paradigm", "Atmosphere is shared memory, vehicles are I/O devices", "Traffic flow obeys the vehicular equation of state", "Phase transitions: free-flow (gas) \u2192 synchronized (liquid) \u2192 platoon (crystal)"],
    description: "Sango-Rine-Shumba is a distributed internet protocol based on gas thermodynamics. Vehicles communicate not through radio (V2V) but through the atmosphere (V2A2V). Each vehicle modifies the local atmospheric state (exhaust, thermal wake, pressure waves), and other vehicles read these modifications through their membranes. The protocol governs how information propagates, with traffic dynamics following the vehicular equation of state."
  },
  {
    id: "ai", layer: "Layer 7", name: "Zangalewa AI Interceptor",
    tagline: "AI layer between OS and any external system", color: "#58E6D9",
    equations: [],
    keyResults: ["Autonomous command orchestration and error correction", "Knowledge base construction via FAISS vectorized storage", "AST-level code analysis with semantic understanding", "Git-tracked error resolution with automatic rollback", "Metacognitive process manager with context tracking"],
    description: "Zangalewa sits between the Buhera operating system and any external AI or LLM system. It intercepts commands, validates them against the categorical framework, corrects errors, and maintains a knowledge base of the vehicle's operational history. It provides the intelligence layer that bridges the formal TCC computing paradigm with the flexibility of modern AI systems."
  },
  {
    id: "learning", layer: "Layer 8", name: "Federated Multi-Modal Understanding",
    tagline: "Privacy-preserving distributed learning across vehicles", color: "#D4AF37",
    equations: ["I(D; A_Q) \\ll H(D)"],
    keyResults: ["Compression ratio 10\u207B\u00B3 to 10\u207B\u2077 \u2014 vehicles share categorical summaries, not raw data", "Privacy by construction: S-entropy coordinates don't reveal raw sensor content", "All sensor modalities produce points in the same [0,1]\u00B3 \u2014 fusion is averaging", "Fleet-wide learning without centralized data collection"],
    description: "Federated learning in the Vesicle framework operates in S-entropy space. Each vehicle computes local S-entropy coordinates from its sensors and shares only these categorical summaries with the fleet. Because S-entropy coordinates encode what matters for driving (categorical state) without revealing what they measured (raw data), privacy is guaranteed by the mathematics \u2014 not by policy."
  },
  {
    id: "electrical", layer: "Layer 9", name: "Current Flux Mechanism",
    tagline: "Membrane signals converted to usable electrical current", color: "#2AA198",
    equations: ["\\mathbf{J} = \\sigma \\mathbf{E} + D\\nabla n"],
    keyResults: ["Electrical current reinterpreted as categorical state propagation", "Membrane oscillatory states propagate through P-N junction as electrical signals", "Categorical converter: temperature controls effective computational base", "b_eff(T) = 1 + (b_max-1)(1 - e^{-\u0394E/k_BT}), optimal at b_eff = 3 (ternary)"],
    description: "The current flux mechanism explains how the membrane's categorical information becomes usable electrical signals. Current is not electron drift but categorical state propagation \u2014 each oscillatory state in the membrane P-N junction propagates as a categorical address, carrying information at the speed of the partition dynamics rather than the drift velocity of electrons."
  },
];

export default function Platform() {
  const [expandedId, setExpandedId] = useState(null);

  return (
    <>
      <Head>
        <title>Platform | Vesicle</title>
        <meta name="description" content="The complete 9-layer Vesicle autonomous vehicle platform." />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        <Layout>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-4xl mx-auto">
            <h1 className="text-5xl font-bold text-center mb-4 md:text-4xl">The Vesicle Platform</h1>
            <p className="text-light/50 text-center mb-16 text-lg">
              Nine layers. One axiom. Zero free parameters.
            </p>

            <div className="space-y-3">
              {vesicleStack.map((layer, i) => (
                <motion.div
                  key={layer.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="border border-light/10 rounded-xl overflow-hidden"
                >
                  <button
                    onClick={() => setExpandedId(expandedId === layer.id ? null : layer.id)}
                    className="w-full flex items-center gap-4 p-5 text-left hover:bg-light/5 transition-colors"
                  >
                    <span className="text-xs font-mono px-2 py-1 rounded" style={{ backgroundColor: layer.color + "20", color: layer.color }}>
                      {layer.layer}
                    </span>
                    <div className="flex-1">
                      <div className="font-bold text-lg">{layer.name}</div>
                      <div className="text-sm text-light/50">{layer.tagline}</div>
                    </div>
                    <span className="text-light/30 text-xl">{expandedId === layer.id ? "\u2212" : "+"}</span>
                  </button>

                  <AnimatePresence>
                    {expandedId === layer.id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                      >
                        <div className="px-5 pb-5 border-t border-light/5 pt-4">
                          <p className="text-light/70 mb-4 leading-relaxed">{layer.description}</p>

                          {layer.equations.length > 0 && (
                            <div className="mb-4 bg-light/5 rounded-lg p-4">
                              <div className="text-xs text-light/40 uppercase tracking-wider mb-2">Key Equations</div>
                              {layer.equations.map((eq, j) => (
                                <div key={j} className="my-2">
                                  <DisplayEquation math={eq} />
                                </div>
                              ))}
                            </div>
                          )}

                          <div className="text-xs text-light/40 uppercase tracking-wider mb-2">Key Results</div>
                          <ul className="space-y-1">
                            {layer.keyResults.map((r, j) => (
                              <li key={j} className="text-sm text-light/60 flex gap-2">
                                <span style={{ color: layer.color }}>&#x2022;</span>
                                {r}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

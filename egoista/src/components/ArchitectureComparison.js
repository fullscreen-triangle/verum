// components/ArchitectureComparison.js
import { useRef } from "react";
import { useInView } from "framer-motion";

export default function ArchitectureComparison() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  const comparison = [
    {
      aspect: "Computational Paradigm",
      conventional: "Forward simulation (predict future states)",
      trajectory: "Backward completion (navigate to penultimate state)",
      advantage: "10³–10⁵× speedup"
    },
    {
      aspect: "Prediction Horizon",
      conventional: "5–15 seconds required, 0.5–2 seconds achievable",
      trajectory: "No prediction needed (sufficiency recognition)",
      advantage: "Eliminates Lyapunov divergence"
    },
    {
      aspect: "Information Processing",
      conventional: "10⁸–10⁹ bits/frame (full sensor data)",
      trajectory: "2–5 bits/decision (mutual information only)",
      advantage: "10⁷× compression ratio"
    },
    {
      aspect: "Architecture",
      conventional: "Perceive → Predict → Plan → Control (serial pipeline)",
      trajectory: "Unified counting loop network (parallel coupling)",
      advantage: "No pipeline fragility"
    },
    {
      aspect: "Safety Guarantee",
      conventional: "Probabilistic (Pₓₐᵢₗ → 1 as t → ∞)",
      trajectory: "Deterministic (dcat ≤ √3, ε > 0 ensures recognition)",
      advantage: "Mathematical proof"
    },
    {
      aspect: "Occlusion Handling",
      conventional: "Fails (photon-dependent sensors)",
      trajectory: "Immune (∂dcat/∂τoptical = 0)",
      advantage: "Works in fog/darkness"
    },
    {
      aspect: "Security",
      conventional: "Cryptographic (vulnerable to quantum computing)",
      trajectory: "Thermodynamic (attackers violate Second Law)",
      advantage: "Physically unforgeable"
    }
  ];

  return (
    <div ref={ref} className="my-32">
      <h2 className="font-bold text-8xl mb-16 w-full text-center md:text-6xl xs:text-4xl md:mb-8">
        Architecture Comparison
      </h2>
      
      <div className={`relative rounded-2xl border-2 border-solid border-dark bg-light dark:border-light dark:bg-dark transition-all duration-700 ${
        isInView ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
      }`}>
        <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-dark dark:border-light">
                <th className="p-4 text-left font-bold text-lg md:text-base">Aspect</th>
                <th className="p-4 text-left font-bold text-lg md:text-base">Conventional AV</th>
                <th className="p-4 text-left font-bold text-lg md:text-base">Trajectory Completion</th>
                <th className="p-4 text-left font-bold text-lg md:text-base text-primary dark:text-primaryDark">Advantage</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row, index) => (
                <tr 
                  key={index} 
                  className={`border-b border-dark/20 dark:border-light/20 transition-all duration-500 ${
                    isInView ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-5'
                  }`}
                  style={{ transitionDelay: `${index * 100}ms` }}
                >
                  <td className="p-4 font-semibold md:text-sm">{row.aspect}</td>
                  <td className="p-4 text-dark/75 dark:text-light/75 md:text-sm">{row.conventional}</td>
                  <td className="p-4 text-dark/75 dark:text-light/75 md:text-sm">{row.trajectory}</td>
                  <td className="p-4 font-semibold text-primary dark:text-primaryDark md:text-sm">{row.advantage}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

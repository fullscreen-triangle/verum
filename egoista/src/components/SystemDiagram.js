// components/SystemDiagram.js
import { useRef } from "react";
import { useInView } from "framer-motion";

export default function SystemDiagram() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  return (
    <div ref={ref} className="my-32">
      <h2 className="font-bold text-8xl mb-16 w-full text-center md:text-6xl xs:text-4xl md:mb-8">
        System Architecture
      </h2>
      
      <div className={`relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark transition-all duration-700 ${
        isInView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
      }`}>
        <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
        
        {/* SVG Diagram */}
        <svg viewBox="0 0 800 600" className="w-full h-auto">
          {/* Observer Counting Loops */}
          <g>
            <circle cx="100" cy="100" r="40" fill="#3b82f6" opacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="100" y="105" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">Wheel</text>
            <text x="100" y="120" textAnchor="middle" fill="currentColor" fontSize="10">10 Hz</text>
          </g>
          
          <g>
            <circle cx="200" cy="100" r="40" fill="#3b82f6" opacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="200" y="105" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">Camera</text>
            <text x="200" y="120" textAnchor="middle" fill="currentColor" fontSize="10">60 Hz</text>
          </g>
          
          <g>
            <circle cx="300" cy="100" r="40" fill="#3b82f6" opacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="300" y="105" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">LiDAR</text>
            <text x="300" y="120" textAnchor="middle" fill="currentColor" fontSize="10">20 Hz</text>
          </g>
          
          <g>
            <circle cx="400" cy="100" r="40" fill="#3b82f6" opacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="400" y="105" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">IMU</text>
            <text x="400" y="120" textAnchor="middle" fill="currentColor" fontSize="10">1 kHz</text>
          </g>
          
          <g>
            <circle cx="500" cy="100" r="40" fill="#3b82f6" opacity="0.2" stroke="#3b82f6" strokeWidth="2" />
            <text x="500" y="105" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">Engine</text>
            <text x="500" y="120" textAnchor="middle" fill="currentColor" fontSize="10">50 Hz</text>
          </g>

          {/* CSM */}
          <rect x="150" y="200" width="300" height="80" rx="10" fill="#10b981" opacity="0.2" stroke="#10b981" strokeWidth="2" />
          <text x="300" y="230" textAnchor="middle" fill="currentColor" fontSize="16" fontWeight="bold">Categorical State Manager</text>
          <text x="300" y="250" textAnchor="middle" fill="currentColor" fontSize="12">S = (Sk, St, Se) ∈ [0,1]³</text>
          <text x="300" y="265" textAnchor="middle" fill="currentColor" fontSize="10">Inverse-variance weighting</text>

          {/* Arrows from observers to CSM */}
          <line x1="100" y1="140" x2="200" y2="200" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowblue)" />
          <line x1="200" y1="140" x2="250" y2="200" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowblue)" />
          <line x1="300" y1="140" x2="300" y2="200" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowblue)" />
          <line x1="400" y1="140" x2="350" y2="200" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowblue)" />
          <line x1="500" y1="140" x2="400" y2="200" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowblue)" />

          {/* PNE and SRM */}
          <rect x="50" y="330" width="200" height="80" rx="10" fill="#f59e0b" opacity="0.2" stroke="#f59e0b" strokeWidth="2" />
          <text x="150" y="360" textAnchor="middle" fill="currentColor" fontSize="14" fontWeight="bold">Penultimate Navigation</text>
          <text x="150" y="380" textAnchor="middle" fill="currentColor" fontSize="12">O(log₃ N) backward</text>
          <text x="150" y="395" textAnchor="middle" fill="currentColor" fontSize="10">completion</text>

          <rect x="350" y="330" width="200" height="80" rx="10" fill="#f59e0b" opacity="0.2" stroke="#f59e0b" strokeWidth="2" />
          <text x="450" y="360" textAnchor="middle" fill="currentColor" fontSize="14" fontWeight="bold">Sufficiency Recognition</text>
          <text x="450" y="380" textAnchor="middle" fill="currentColor" fontSize="12">Triple convergence</text>
          <text x="450" y="395" textAnchor="middle" fill="currentColor" fontSize="10">εosc ≈ εcat ≈ εpar</text>

          {/* Arrows from CSM */}
          <line x1="250" y1="280" x2="150" y2="330" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrowgreen)" />
          <line x1="350" y1="280" x2="450" y2="330" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrowgreen)" />

          {/* CME */}
          <rect x="200" y="460" width="200" height="80" rx="10" fill="#ef4444" opacity="0.2" stroke="#ef4444" strokeWidth="2" />
          <text x="300" y="490" textAnchor="middle" fill="currentColor" fontSize="14" fontWeight="bold">Completion Morphism</text>
          <text x="300" y="510" textAnchor="middle" fill="currentColor" fontSize="12">Spen → Sfinal</text>
          <text x="300" y="525" textAnchor="middle" fill="currentColor" fontSize="10">Physical coupling</text>

          {/* Arrows to CME */}
          <line x1="150" y1="410" x2="250" y2="460" stroke="#f59e0b" strokeWidth="2" markerEnd="url(#arroworange)" />
          <line x1="450" y1="410" x2="350" y2="460" stroke="#f59e0b" strokeWidth="2" markerEnd="url(#arroworange)" />

          {/* TEM */}
          <rect x="600" y="250" width="150" height="120" rx="10" fill="#8b5cf6" opacity="0.2" stroke="#8b5cf6" strokeWidth="2" />
          <text x="675" y="280" textAnchor="middle" fill="currentColor" fontSize="14" fontWeight="bold">Triple Equivalence</text>
          <text x="675" y="300" textAnchor="middle" fill="currentColor" fontSize="14" fontWeight="bold">Monitor</text>
          <text x="675" y="325" textAnchor="middle" fill="currentColor" fontSize="11">dM/dt = ω/(2π/M)</text>
          <text x="675" y="345" textAnchor="middle" fill="currentColor" fontSize="11">= 1/⟨τp⟩</text>

          {/* Monitoring arrows */}
          <line x1="600" y1="310" x2="450" y2="240" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="5,5" />

          {/* Arrow markers */}
          <defs>
            <marker id="arrowblue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#3b82f6" />
            </marker>
            <marker id="arrowgreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#10b981" />
            </marker>
            <marker id="arroworange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#f59e0b" />
            </marker>
          </defs>
        </svg>

        <p className="mt-6 text-sm text-dark/75 dark:text-light/75 font-medium text-center">
          Unlike conventional pipelines, information flows through oscillator couplings in shared partition space. 
          All subsystems operate simultaneously as coupled counting loop sub-networks.
        </p>
      </div>
    </div>
  );
}

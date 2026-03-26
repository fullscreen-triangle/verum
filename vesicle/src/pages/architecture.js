// pages/architecture.js
import Layout from "@/components/Layout";
import Head from "next/head";
import { useInView, useMotionValue, useSpring } from "framer-motion";
import { useEffect, useRef } from "react";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import ArchitectureComparison from "@/components/ArchitectureComparison";
import PerformanceMetrics from "@/components/PerformanceMetrics";
import SystemDiagram from "@/components/SystemDiagram";

function AnimatedNumberFramerMotion({ value, suffix = "" }) {
  const ref = useRef(null);
  const motionValue = useMotionValue(0);
  const springValue = useSpring(motionValue, { duration: 3000 });
  const isInView = useInView(ref, { once: true });
  
  useEffect(() => {
    if (isInView) {
      motionValue.set(value);
    }
  }, [motionValue, value, isInView]);

  useEffect(
    () =>
      springValue.on("change", (latest) => {
        if (ref.current && latest.toFixed(0) <= value) {
          ref.current.textContent = latest.toFixed(0) + suffix;
        }
      }),
    [springValue, value, suffix]
  );

  return <span ref={ref} />;
}

export default function Architecture() {
  return (
    <>
      <Head>
        <title>Trajectory Completion Architecture | Autonomous Vehicles</title>
        <meta 
          name="description" 
          content="Revolutionary autonomous vehicle architecture based on categorical trajectory completion, eliminating prediction-based failures through backward navigation and sufficiency recognition." 
        />
      </Head>
      <TransitionEffect />
      <main className={`flex w-full flex-col items-center justify-center dark:text-light`}>
        <Layout className="pt-16">
          <AnimatedText
            text="Replacing Forward Simulation"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          <div className="grid w-full grid-cols-8 gap-16 sm:gap-8">
            {/* Main Content */}
            <div className="col-span-5 flex flex-col items-start justify-start xl:col-span-8 md:order-2">
              <h2 className="mb-4 text-lg font-bold uppercase text-dark/75 dark:text-light/75">
                THE FUNDAMENTAL PROBLEM
              </h2>
              <p className="font-medium">
                Every autonomous vehicle deployed today follows the same pipeline:{" "}
                <strong>Perceive → Predict → Plan → Control</strong>. This architecture 
                has consumed over $100 billion in investment, yet it is mathematically 
                guaranteed to fail.
              </p>
              <p className="my-4 font-medium">
                The failure is not engineering immaturity—it's <strong>mathematical impossibility</strong>. 
                Lyapunov divergence renders prediction useless in 0.5–2 seconds for highway traffic. 
                Information-theoretic analysis shows prediction wastes 10³–10⁷× more computation 
                than driving actually requires.
              </p>
              <p className="font-medium">
                We present a fundamentally different architecture based on{" "}
                <strong>trajectory completion</strong>: the vehicle navigates backward 
                through partition space at <strong>O(log₃ N)</strong> complexity, eliminating 
                prediction entirely. Driving decisions reduce to <strong>sufficiency recognition</strong> 
                via triple convergence—no simulation, no planning module, no distinction between 
                computer and car.
              </p>

              <h2 className="mt-8 mb-4 text-lg font-bold uppercase text-dark/75 dark:text-light/75">
                KEY INNOVATIONS
              </h2>
              <ul className="list-disc ml-6 space-y-2 font-medium">
                <li>
                  <strong>Counting Loop Primitive:</strong> Observation ≡ Computation ≡ Processing 
                  (O ≡ C ≡ P) unified in a single tick operation
                </li>
                <li>
                  <strong>Backward Navigation:</strong> O(log₃ N) complexity vs. O(N) forward 
                  simulation, 10³–10⁵× speedup on real road networks
                </li>
                <li>
                  <strong>Sufficiency Recognition:</strong> Triple convergence (oscillatory, 
                  categorical, partition) replaces probabilistic prediction
                </li>
                <li>
                  <strong>Coupled Oscillator Network:</strong> Vehicle components (engine, wheels, 
                  sensors, CPU) form unified computational substrate
                </li>
                <li>
                  <strong>Thermodynamic Security:</strong> Unauthorized transitions produce 
                  detectable heating (ΔS &gt; 0), unforgeable by attackers
                </li>
              </ul>
            </div>

            {/* Stats Sidebar */}
            <div className="col-span-3 flex flex-col items-end justify-start xl:col-span-8 xl:flex-row xl:items-center md:order-1 gap-8">
              <div className="w-full relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark">
                <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
                
                <div className="flex flex-col items-center justify-center mb-6">
                  <span className="inline-block text-7xl font-bold md:text-6xl sm:text-5xl xs:text-4xl text-primary dark:text-primaryDark">
                    O(log₃ N)
                  </span>
                  <h2 className="text-xl font-medium capitalize text-dark/75 dark:text-light/75 text-center md:text-lg sm:text-base xs:text-sm">
                    Navigation Complexity
                  </h2>
                </div>

                <div className="flex flex-col items-center justify-center mb-6">
                  <span className="inline-block text-7xl font-bold md:text-6xl sm:text-5xl xs:text-4xl text-primary dark:text-primaryDark">
                    <AnimatedNumberFramerMotion value={2.3} />s
                  </span>
                  <h2 className="text-xl font-medium capitalize text-dark/75 dark:text-light/75 text-center md:text-lg sm:text-base xs:text-sm">
                    Prediction Horizon Limit
                  </h2>
                </div>

                <div className="flex flex-col items-center justify-center">
                  <span className="inline-block text-7xl font-bold md:text-6xl sm:text-5xl xs:text-4xl text-primary dark:text-primaryDark">
                    <AnimatedNumberFramerMotion value={10} />⁵×
                  </span>
                  <h2 className="text-xl font-medium capitalize text-dark/75 dark:text-light/75 text-center md:text-lg sm:text-base xs:text-sm">
                    Speedup vs. A*
                  </h2>
                </div>
              </div>
            </div>
          </div>

          {/* Performance Comparison Chart */}
          <PerformanceMetrics />

          {/* Architecture Comparison */}
          <ArchitectureComparison />

          {/* System Diagram */}
          <SystemDiagram />

          {/* Five Subsystems */}
          <FiveSubsystems />

          {/* Download Paper */}
          <div className="my-32 w-full flex flex-col items-center">
            <h2 className="font-bold text-6xl mb-8 w-full text-center md:text-4xl xs:text-3xl">
              Read The Full Paper
            </h2>
            <a
              href="/papers/autonomous_vehicle_system_architecture.pdf"
              target="_blank"
              className="flex items-center rounded-lg border-2 border-solid bg-dark p-6 px-12 text-lg font-semibold capitalize text-light hover:border-dark hover:bg-transparent hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light dark:hover:bg-dark dark:hover:text-light md:p-4 md:px-8 md:text-base transition-all duration-300"
            >
              Download PDF
            </a>
          </div>
        </Layout>
      </main>
    </>
  );
}

// Five Subsystems Component
function FiveSubsystems() {
  const subsystems = [
    {
      name: "Categorical State Manager (CSM)",
      description: "Maintains current S-entropy triple S = (Sk, St, Se) from all Observer counting loops",
      features: [
        "Inverse-variance weighting for multi-sensor fusion",
        "Convergence error ≤ C/√K with K observers",
        "Urgency tier assignment based on categorical distance"
      ]
    },
    {
      name: "Penultimate Navigation Engine (PNE)",
      description: "Computes penultimate state via backward trajectory completion through ternary partition tree",
      features: [
        "O(log₃ N) complexity on road networks",
        "Returns unique cell one morphism from destination",
        "No forward simulation required"
      ]
    },
    {
      name: "Sufficiency Recognition Module (SRM)",
      description: "Evaluates transition safety via triple convergence of oscillatory, categorical, and partition perspectives",
      features: [
        "Returns Proceed/Slow/Stop decisions",
        "Gödelian safety: ε > 0 ensures recognition",
        "No probabilistic prediction"
      ]
    },
    {
      name: "Completion Morphism Executor (CME)",
      description: "Realizes single-step transition Sₚₑₙ → Sₓᵢₙₐₗ through coupled oscillator network",
      features: [
        "Not a controller—physical coupling topology itself",
        "Kuramoto synchronization for harmonic coincidence",
        "Direct actuation via oscillator phase-locking"
      ]
    },
    {
      name: "Triple Equivalence Monitor (TEM)",
      description: "Continuously verifies rate equation dM/dt = ω/(2π/M) = 1/⟨τₚ⟩ for all counting loops",
      features: [
        "Detects violations within one tick period",
        "Thermodynamic security via entropy monitoring",
        "Unforgeable by attackers (Second Law enforcement)"
      ]
    }
  ];

  return (
    <div className="my-32">
      <h2 className="font-bold text-8xl mb-16 w-full text-center md:text-6xl xs:text-4xl md:mb-8">
        Five Subsystems
      </h2>
      <div className="grid grid-cols-1 gap-8 md:gap-6">
        {subsystems.map((system, index) => (
          <SubsystemCard key={index} {...system} index={index} />
        ))}
      </div>
    </div>
  );
}

function SubsystemCard({ name, description, features, index }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  return (
    <div
      ref={ref}
      className={`relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark transition-all duration-500 ${
        isInView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
      }`}
      style={{ transitionDelay: `${index * 100}ms` }}
    >
      <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
      
      <h3 className="text-2xl font-bold mb-4 text-primary dark:text-primaryDark md:text-xl sm:text-lg">
        {name}
      </h3>
      <p className="font-medium mb-4 text-dark/75 dark:text-light/75">
        {description}
      </p>
      <ul className="list-disc ml-6 space-y-2">
        {features.map((feature, i) => (
          <li key={i} className="font-medium text-sm">
            {feature}
          </li>
        ))}
      </ul>
    </div>
  );
}

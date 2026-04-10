import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import Link from "next/link";
import dynamic from "next/dynamic";

const DisplayEquation = dynamic(() => import("@/components/KatexBlock"), { ssr: false });
const InlineEq = dynamic(
  () => import("@/components/KatexBlock").then((mod) => mod.InlineEquation),
  { ssr: false }
);

const fadeInUp = {
  initial: { y: 50, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

const chainSteps = [
  {
    title: "Bounded Phase Space",
    description: "All physical systems occupy finite volume in phase space.",
    equation: "V_\\Gamma < \\infty",
    explanation:
      "This is the single axiom. No physical system explores infinite phase space volume. This is physically obvious (finite energy, finite volume) but mathematically powerful: it guarantees measure-preserving dynamics via Liouville's theorem.",
  },
  {
    title: "Poincar\u00e9 Recurrence",
    description: "Finite phase space guarantees every trajectory returns.",
    equation: "\\tau_{\\text{rec}} < \\infty",
    explanation:
      "By the Poincar\u00e9 recurrence theorem, any trajectory in a bounded measure-preserving system returns arbitrarily close to its initial state in finite time. This is not an assumption -- it is a mathematical consequence of the axiom.",
  },
  {
    title: "Oscillatory Dynamics",
    description: "Recurrence implies all dynamics are fundamentally oscillatory.",
    equation: "\\frac{dM}{dt} = \\frac{\\omega}{2\\pi / M} = \\frac{1}{\\langle \\tau_p \\rangle}",
    explanation:
      "The fundamental identity. The rate of counting operations M equals the oscillation frequency divided by the period-per-count, which equals the inverse mean partition crossing time. All three are identical -- this is not a model but a tautology of bounded dynamics.",
  },
  {
    title: "Categorical States",
    description: "Oscillatory dynamics partition phase space into countable categories.",
    equation: "C(n) = 2n^2",
    explanation:
      "The capacity formula. A system with quantum number n supports exactly 2n\u00B2 categorical states. This reproduces electron shell structure, molecular orbital theory, and spectroscopic selection rules -- all from counting, with zero adjustable parameters.",
  },
  {
    title: "Triple Equivalence",
    description: "Observation, computation, and processing become the same operation.",
    equation: "O(x) \\equiv C(x) \\equiv P(x)",
    explanation:
      "Observation (O), computation (C), and processing (P) are mathematically identical operations: categorical address resolution. When a membrane observes a molecule, it computes its categorical state. When it computes, it processes. The three cannot be separated.",
  },
  {
    title: "S-Entropy Coordinates",
    description: "S = k_B M ln n provides a universal coordinate system from counting.",
    equation: "S = k_B \\, M \\ln n",
    explanation:
      "Entropy is not disorder. It is the number of counting operations M on n categorical states, scaled by Boltzmann's constant. This reproduces the Boltzmann entropy formula but is derived from first principles -- not postulated. Every physical quantity maps to a point in (S_k, S_t, S_e) space.",
  },
  {
    title: "Backward Trajectory Completion",
    description: "Navigate backward through partition space at O(log\u2083 N) complexity.",
    equation: "\\text{BTC}: O(\\log_3 N) \\text{ vs forward simulation: } O(N)",
    explanation:
      "Instead of simulating the world forward (exponentially chaotic), BTC navigates backward from a declared terminal state through the ternary partition hierarchy. Each step eliminates 2/3 of possibilities. For 10\u2076 states, this is ~13 steps instead of 10\u2076.",
  },
  {
    title: "Membrane Computing",
    description: "Biological membranes implement counting loops as computational substrate.",
    equation: "R_{\\text{total}} = \\frac{2A}{A_L} \\times f_{\\text{iso}} \\approx 10^{28} \\text{ ops/s}",
    explanation:
      "A vehicle surface (~10 m\u00B2) covered in lipid membrane processes at 10\u00B2\u2078 operations per second. Each lipid oscillates at 10\u00B9\u00B9 Hz. By the oscillator-processor duality, each oscillation IS a computation. This is not metaphor -- it is the triple equivalence identity applied to lipids.",
  },
];

const keyResults = [
  {
    formula: "S = k_B \\, M \\ln n",
    title: "Entropy from Counting",
    description:
      "Entropy is not disorder. It is the number of counting operations M on n categorical states. Derived, not assumed.",
  },
  {
    formula: "O(\\log_3 N)",
    title: "Navigation Complexity",
    description:
      "Backward trajectory completion through ternary partition trees gives logarithmic navigation, not linear forward simulation.",
  },
  {
    formula: "\\lambda_{\\text{partition}} = 0",
    title: "No Chaos",
    description:
      "The partition Lyapunov exponent is exactly zero. Categorical dynamics are inherently stable -- no butterfly effect.",
  },
  {
    formula: "[\\hat{O}_{\\text{cat}}, \\hat{O}_{\\text{phys}}] = 0",
    title: "Zero Backaction",
    description:
      "The categorical observer commutes with the physical observable. Measurement does not disturb the system.",
  },
];

const transportEquations = [
  {
    equation: "\\Xi = N^{-1} \\sum_{i,j} \\tau_{p,ij} \\, g_{ij}",
    title: "Transport Formula",
    description:
      "The universal transport coefficient. Every transport phenomenon (diffusion, viscosity, thermal conductivity, electrical conductivity) is expressed as a weighted sum of partition crossing times and metric factors. No phenomenological constants -- all derived from the axiom.",
  },
  {
    equation: "\\Pi: \\mathbb{R}^3 \\to [0,1]^3",
    title: "Position-Partition Bijection",
    description:
      "The Cynegeticus map. Every point in physical space maps to a unique point in S-entropy space. The Jacobian is non-singular everywhere (proven), so the inverse map recovers position from atmospheric molecular state. GPS without satellites.",
  },
  {
    equation: "P_{\\text{drive}} \\cdot V_{\\text{road}} = N \\cdot k_B \\cdot T_{\\text{cat}}",
    title: "Vehicular Equation of State",
    description:
      "Traffic flow obeys an ideal gas law. Drive pressure times road volume equals vehicle count times categorical temperature. Phase transitions: free-flow (gas), synchronized (liquid), platoon (crystal). No free parameters.",
  },
];

export default function Framework() {
  return (
    <>
      <Head>
        <title>Framework | Vesicle</title>
        <meta
          name="description"
          content="From one axiom to autonomous driving. The complete theoretical framework behind the Vesicle membrane computing architecture."
        />
      </Head>
      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="From One Axiom to Autonomous Driving"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          {/* Section 1: The Axiom */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              The Axiom
            </h2>
            <div className="relative rounded-2xl border-2 border-solid border-gold bg-light p-12 dark:border-gold dark:bg-dark text-center">
              <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-gold/20 dark:bg-gold/10" />
              <div className="text-4xl font-bold text-primary dark:text-primaryDark mb-6 md:text-3xl sm:text-2xl xs:text-xl">
                <DisplayEquation math="V_\\Gamma < \\infty" />
              </div>
              <p className="text-2xl font-medium text-dark/75 dark:text-light/75 md:text-xl sm:text-lg">
                All physical systems occupy finite phase space.
              </p>
              <p className="text-lg font-medium text-dark/60 dark:text-light/60 mt-4 max-w-2xl mx-auto md:text-base">
                This single statement -- that no physical system explores infinite phase space volume
                -- is the only assumption. Everything else is derived. No fitting, no parameters, no
                approximations. One axiom produces the entire framework.
              </p>
            </div>
          </motion.div>

          {/* Section 2: The Chain */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              The Derivation Chain
            </h2>
            <div className="grid grid-cols-1 gap-6">
              {chainSteps.map((step, index) => (
                <motion.div
                  key={index}
                  initial={{ y: 30, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="relative flex items-start gap-6 md:gap-4"
                >
                  {/* Step Number + Connector */}
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 rounded-full bg-primary dark:bg-primaryDark flex items-center justify-center text-light dark:text-dark font-bold text-lg shrink-0">
                      {index + 1}
                    </div>
                    {index < chainSteps.length - 1 && (
                      <div className="w-0.5 h-8 bg-primary/30 dark:bg-primaryDark/30 mt-2" />
                    )}
                  </div>
                  {/* Content */}
                  <div className="pb-6 flex-1">
                    <h3 className="text-2xl font-bold mb-2 text-dark dark:text-light md:text-xl sm:text-lg">
                      {step.title}
                    </h3>
                    <div className="bg-dark/5 dark:bg-light/5 rounded-lg p-4 mb-3">
                      <DisplayEquation math={step.equation} />
                    </div>
                    <p className="font-medium text-dark/75 dark:text-light/75 md:text-sm leading-relaxed">
                      {step.explanation}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Section 3: The Identity */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              The Identity
            </h2>
            <div className="relative rounded-2xl border-2 border-solid border-dark bg-light p-12 dark:border-light dark:bg-dark text-center">
              <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
              <div className="text-5xl font-bold text-primary dark:text-primaryDark mb-6 md:text-4xl sm:text-3xl xs:text-2xl">
                <DisplayEquation math="O(x) \\equiv C(x) \\equiv P(x)" />
              </div>
              <p className="text-xl font-medium text-dark/75 dark:text-light/75 max-w-3xl mx-auto md:text-lg sm:text-base">
                Observation, computation, and processing are the same operation. This is not a
                metaphor or an analogy -- it is a mathematical identity. When a membrane observes a
                molecule, it computes. When it computes, it processes. The three cannot be
                separated. This is why the membrane IS the computer, and why the vehicle IS its own
                navigation system.
              </p>
            </div>
          </motion.div>

          {/* Section 4: Key Results */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              Key Results
            </h2>
            <div className="grid grid-cols-2 gap-8 md:grid-cols-1 md:gap-6">
              {keyResults.map((result, index) => (
                <motion.div
                  key={index}
                  initial={{ y: 40, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark"
                >
                  <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
                  <div className="text-2xl font-bold text-primary dark:text-primaryDark mb-4 md:text-xl">
                    <DisplayEquation math={result.formula} />
                  </div>
                  <h3 className="text-xl font-bold mb-3 text-dark dark:text-light md:text-lg">
                    {result.title}
                  </h3>
                  <p className="font-medium text-dark/75 dark:text-light/75 md:text-sm">
                    {result.description}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Section 5: Transport & Positioning */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              Transport & Positioning
            </h2>
            <div className="grid grid-cols-1 gap-8">
              {transportEquations.map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ y: 30, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="relative rounded-2xl border border-solid border-dark/20 bg-light p-8 dark:border-light/20 dark:bg-dark"
                >
                  <div className="bg-dark/5 dark:bg-light/5 rounded-lg p-4 mb-4">
                    <DisplayEquation math={item.equation} />
                  </div>
                  <h3 className="text-xl font-bold mb-3 text-dark dark:text-light md:text-lg">
                    {item.title}
                  </h3>
                  <p className="font-medium text-dark/75 dark:text-light/75 md:text-sm leading-relaxed">
                    {item.description}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Link to Papers */}
          <div className="w-full flex flex-col items-center mb-16">
            <p className="text-xl font-medium text-dark/75 dark:text-light/75 mb-8 text-center md:text-lg">
              Explore the complete publication portfolio behind the framework.
            </p>
            <Link
              href="/papers"
              className="flex items-center rounded-lg border-2 border-solid bg-dark p-6 px-12 text-lg font-semibold capitalize text-light hover:border-dark hover:bg-transparent hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light dark:hover:bg-dark dark:hover:text-light md:p-4 md:px-8 md:text-base transition-all duration-300"
            >
              View All Papers
            </Link>
          </div>
        </Layout>
      </main>
    </>
  );
}

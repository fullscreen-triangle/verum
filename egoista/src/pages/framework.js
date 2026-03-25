import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import Link from "next/link";

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
  },
  {
    title: "Poincar\u00e9 Recurrence",
    description: "Finite phase space guarantees every trajectory returns.",
  },
  {
    title: "Oscillatory Dynamics",
    description: "Recurrence implies all dynamics are fundamentally oscillatory.",
  },
  {
    title: "Categorical States",
    description: "Oscillatory dynamics partition phase space into countable categories.",
  },
  {
    title: "Triple Equivalence",
    description: "Observation, computation, and processing become the same operation.",
  },
  {
    title: "S-Entropy Coordinates",
    description: "S = k_B M ln n provides a universal coordinate system from counting.",
  },
  {
    title: "Backward Trajectory Completion",
    description: "Navigate backward through partition space at O(log\u2083 N) complexity.",
  },
  {
    title: "Membrane Computing",
    description: "Biological membranes implement counting loops as computational substrate.",
  },
];

const keyResults = [
  {
    formula: "S = k_B M ln n",
    title: "Entropy from Counting",
    description:
      "Entropy is not disorder. It is the number of counting operations M on n categorical states. Derived, not assumed.",
  },
  {
    formula: "O(log\u2083 N)",
    title: "Navigation Complexity",
    description:
      "Backward trajectory completion through ternary partition trees gives logarithmic navigation, not linear forward simulation.",
  },
  {
    formula: "\u03bb_partition = 0",
    title: "No Chaos",
    description:
      "The partition Lyapunov exponent is exactly zero. Categorical dynamics are inherently stable -- no butterfly effect.",
  },
  {
    formula: "[\u00d4_cat, \u00d4_phys] = 0",
    title: "Zero Backaction",
    description:
      "The categorical observer commutes with the physical observable. Measurement does not disturb the system.",
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
              <p className="text-4xl font-bold text-primary dark:text-primaryDark mb-6 md:text-3xl sm:text-2xl xs:text-xl font-mono">
                V_&#915; &lt; &#8734;
              </p>
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
              The Chain
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
                  <div className="pb-6">
                    <h3 className="text-2xl font-bold mb-2 text-dark dark:text-light md:text-xl sm:text-lg">
                      {step.title}
                    </h3>
                    <p className="font-medium text-dark/75 dark:text-light/75 md:text-sm">
                      {step.description}
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
              <p className="text-5xl font-bold text-primary dark:text-primaryDark mb-6 md:text-4xl sm:text-3xl xs:text-2xl font-mono">
                O(x) &#8801; C(x) &#8801; P(x)
              </p>
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
                  <p className="text-3xl font-bold text-primary dark:text-primaryDark mb-4 font-mono md:text-2xl sm:text-xl">
                    {result.formula}
                  </p>
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

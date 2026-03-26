import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";

const fadeInUp = {
  initial: { y: 50, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

const categories = [
  {
    name: "Core Theory",
    color: "text-primary dark:text-primaryDark",
    papers: [
      {
        title: "Trajectory Completion",
        summary:
          "Foundation paper deriving backward navigation from bounded phase space axiom.",
      },
      {
        title: "Backward Trajectory Completion",
        summary:
          "O(log\u2083 N) navigation through ternary partition trees without forward simulation.",
      },
      {
        title: "Poincar\u00e9 Computing",
        summary:
          "Recurrence theorem as computational primitive -- every trajectory returns.",
      },
      {
        title: "Single-Particle Gas Laws",
        summary:
          "Thermodynamic laws derived for single particles via categorical counting.",
      },
      {
        title: "Gas Ensemble Trajectory Completion",
        summary:
          "Extension of trajectory completion to N-body systems via ensemble categories.",
      },
    ],
  },
  {
    name: "Applications",
    color: "text-gold dark:text-gold",
    papers: [
      {
        title: "Atmospheric Trajectory Completion",
        summary:
          "Weather dynamics as categorical trajectories -- atmospheric S-entropy coordinates.",
      },
      {
        title: "Cynegeticus Positioning",
        summary:
          "Topological positioning system replacing GPS with categorical state recognition.",
      },
      {
        title: "Current-Flux Mechanism",
        summary:
          "Electromagnetic phenomena derived from counting loop rate equations.",
      },
      {
        title: "Mass Transfer",
        summary:
          "Diffusion and transport as backward trajectory completion through concentration partitions.",
      },
      {
        title: "Partition Counting",
        summary:
          "Fundamental counting operation on categorical partitions -- the atomic computation.",
      },
      {
        title: "Partition Depth Limits",
        summary:
          "Maximum refinement depth of ternary partition trees and resolution bounds.",
      },
    ],
  },
  {
    name: "Computing Systems",
    color: "text-membrane dark:text-membrane",
    papers: [
      {
        title: "Buhera OS",
        summary:
          "Operating system built on categorical state management and counting loop scheduling.",
      },
      {
        title: "vaHera Language",
        summary:
          "Programming language where every operation is a counting loop on categorical states.",
      },
      {
        title: "Zangalewa Intent Navigation",
        summary:
          "Navigation system using intent recognition via triple convergence.",
      },
      {
        title: "OberScript",
        summary:
          "Scripting layer for membrane computing with S-entropy type system.",
      },
      {
        title: "Sango Network Protocols",
        summary:
          "Network communication via categorical morphisms -- unforgeable by design.",
      },
    ],
  },
  {
    name: "Membrane Architecture",
    color: "text-primary dark:text-primaryDark",
    papers: [
      {
        title: "Biological Membrane Interface",
        summary:
          "Lipid bilayer as computational surface -- 10^28 ops/s from molecular counting.",
      },
      {
        title: "Categorical Processing Unit",
        summary:
          "Hardware architecture replacing ALU with categorical state transition engine.",
      },
      {
        title: "Molecular Dynamics Memory",
        summary:
          "Memory storage via molecular conformational states in membrane substrate.",
      },
      {
        title: "Oscillatory Quantum Computing",
        summary:
          "Quantum computation as special case of oscillatory categorical dynamics.",
      },
      {
        title: "Oscillatory Logic Circuits",
        summary:
          "Logic gates from coupled oscillators -- AND, OR, NOT via phase relationships.",
      },
      {
        title: "Categorical Converter",
        summary:
          "Interface between conventional digital logic and categorical membrane computing.",
      },
      {
        title: "Lipid Membrane Derivation",
        summary:
          "First-principles derivation of lipid bilayer properties from the axiom.",
      },
    ],
  },
  {
    name: "Autonomous Vehicle",
    color: "text-gold dark:text-gold",
    papers: [
      {
        title: "Equations of State",
        summary:
          "Vehicle dynamics as categorical state equations -- no free parameters.",
      },
      {
        title: "Counting Loops",
        summary:
          "Sensor fusion via inverse-variance weighted counting loops across all observers.",
      },
      {
        title: "Computing Architecture",
        summary:
          "Five-subsystem architecture: CSM, PNE, SRM, CME, TEM replacing perception-prediction-planning.",
      },
      {
        title: "Membrane Sensor",
        summary:
          "Lipid membrane sensor array -- weather-enhanced, sub-$500 per unit.",
      },
    ],
  },
];

function PaperCard({ title, summary }) {
  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
      className="relative rounded-xl border border-solid border-dark/20 bg-light p-6 dark:border-light/20 dark:bg-dark hover:border-primary dark:hover:border-primaryDark transition-colors duration-300"
    >
      <h4 className="text-lg font-bold mb-2 text-dark dark:text-light">{title}</h4>
      <p className="font-medium text-sm text-dark/60 dark:text-light/60">{summary}</p>
    </motion.div>
  );
}

export default function Papers() {
  const totalPapers = categories.reduce((sum, cat) => sum + cat.papers.length, 0);

  return (
    <>
      <Head>
        <title>Papers | Egoista</title>
        <meta
          name="description"
          content="Complete publication portfolio for the Egoista membrane computing framework. 30+ papers, zero free parameters."
        />
      </Head>
      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Publication Portfolio"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          {/* Categories */}
          {categories.map((category, catIndex) => (
            <motion.div
              key={catIndex}
              {...fadeInUp}
              transition={{ duration: 0.6, delay: catIndex * 0.1 }}
              className="mb-16 md:mb-12"
            >
              <div className="flex items-center gap-4 mb-8">
                <h2 className={`font-bold text-4xl md:text-3xl sm:text-2xl ${category.color}`}>
                  {category.name}
                </h2>
                <span className="text-lg font-medium text-dark/50 dark:text-light/50">
                  ({category.papers.length} papers)
                </span>
              </div>
              <div className="grid grid-cols-2 gap-6 md:grid-cols-1 md:gap-4">
                {category.papers.map((paper, paperIndex) => (
                  <PaperCard key={paperIndex} title={paper.title} summary={paper.summary} />
                ))}
              </div>
            </motion.div>
          ))}

          {/* Total */}
          <motion.div
            {...fadeInUp}
            className="w-full flex flex-col items-center justify-center mb-16 mt-8"
          >
            <div className="relative rounded-2xl border-2 border-solid border-gold bg-light p-12 dark:border-gold dark:bg-dark text-center w-full max-w-2xl">
              <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-gold/20 dark:bg-gold/10" />
              <p className="text-5xl font-bold text-primary dark:text-primaryDark mb-4 md:text-4xl sm:text-3xl">
                {totalPapers}+ papers
              </p>
              <p className="text-2xl font-medium text-dark/75 dark:text-light/75 md:text-xl sm:text-lg">
                Zero free parameters
              </p>
            </div>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

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
        pdf: "trajectory-completion.pdf",
      },
      {
        title: "Backward Trajectory Completion",
        summary:
          "O(log\u2083 N) navigation through ternary partition trees without forward simulation.",
        pdf: "backward-trajectory-completion.pdf",
      },
      {
        title: "Poincar\u00e9 Computing",
        summary:
          "Recurrence theorem as computational primitive -- every trajectory returns.",
        pdf: "poincare-trajectory-computing.pdf",
      },
      {
        title: "Single-Particle Gas Laws",
        summary:
          "Thermodynamic laws derived for single particles via categorical counting.",
        pdf: "single-particle-gas-laws.pdf",
      },
      {
        title: "Gas Ensemble Trajectory Completion",
        summary:
          "Extension of trajectory completion to N-body systems via ensemble categories.",
        pdf: "gas-ensemble-trajectory-completion.pdf",
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
        pdf: "atmospheric-trajectory-completion.pdf",
      },
      {
        title: "Cynegeticus Positioning",
        summary:
          "Topological positioning system replacing GPS with categorical state recognition.",
        pdf: "cynegeticus-positioning-script.pdf",
      },
      {
        title: "Current-Flux Mechanism",
        summary:
          "Electromagnetic phenomena derived from counting loop rate equations.",
        pdf: "current-flux-mechanism.pdf",
      },
      {
        title: "Mass Transfer Mechanisms",
        summary:
          "Diffusion and transport as backward trajectory completion through concentration partitions.",
        pdf: "mass-transfer-mechanisms.pdf",
      },
      {
        title: "Partition Counting",
        summary:
          "Fundamental counting operation on categorical partitions -- the atomic computation.",
        pdf: "partition-counting.pdf",
      },
      {
        title: "Partition Depth Limits",
        summary:
          "Maximum refinement depth of ternary partition trees and resolution bounds.",
        pdf: "partition-depth-limits.pdf",
      },
    ],
  },
  {
    name: "Computing Systems",
    color: "text-membrane dark:text-membrane",
    papers: [
      {
        title: "Buhera Operating System",
        summary:
          "Operating system built on categorical state management and counting loop scheduling.",
        pdf: "buhera-operating-system.pdf",
      },
      {
        title: "Buhera OS Architecture",
        summary:
          "Detailed architecture of the non-Turing categorical operating system.",
        pdf: "buhera-os-architecture.pdf",
      },
      {
        title: "vaHera Categorical Scripting",
        summary:
          "Programming language where every operation is a counting loop on categorical states.",
        pdf: "vaHera-categorical-scripting.pdf",
      },
      {
        title: "Zangalewa OS-LLM Interceptor",
        summary:
          "Navigation system using intent recognition via triple convergence.",
        pdf: "zangalewa-os-llm-interceptor.pdf",
      },
      {
        title: "Ober Atmospheric Scripting",
        summary:
          "Scripting layer for membrane computing with S-entropy type system.",
        pdf: "ober-atmos-scripting.pdf",
      },
      {
        title: "Transplanckian Sango-Rine-Shumba",
        summary:
          "Network communication via categorical morphisms -- unforgeable by design.",
        pdf: "transplanckian-sango-rine-shumba.pdf",
      },
    ],
  },
  {
    name: "Membrane Architecture",
    color: "text-primary dark:text-primaryDark",
    papers: [
      {
        title: "Emission Strobe Spectroscopy",
        summary:
          "Spectroscopic technique for reading oscillatory states from membrane surfaces.",
        pdf: "emission-strobe-spectroscopy.pdf",
      },
      {
        title: "Instrument Derivation",
        summary:
          "First-principles derivation of membrane instrumentation from the axiom.",
        pdf: "instrument-derivation.pdf",
      },
      {
        title: "Trajectory Mechanism",
        summary:
          "Mechanism of trajectory completion at the molecular level.",
        pdf: "trajectory-mechanism.pdf",
      },
      {
        title: "Transport Dynamics",
        summary:
          "Transport dynamics and partition depth limits in membrane systems.",
        pdf: "transport-dynamics-partition-limits.pdf",
      },
      {
        title: "Purpose Partition Models",
        summary:
          "Models for purpose-driven partition selection in categorical computing.",
        pdf: "purpose-partition-models.pdf",
      },
    ],
  },
  {
    name: "Integration & Governance",
    color: "text-gold dark:text-gold",
    papers: [
      {
        title: "Federated Multi-Modal Understanding",
        summary:
          "Privacy-preserving distributed learning across membrane vehicles via S-entropy compression.",
        pdf: "federated-multi-modal-understanding.pdf",
      },
      {
        title: "Union of Two Crowns",
        summary:
          "Governance framework for integrating membrane computing with existing infrastructure.",
        pdf: "union-of-two-crowns.pdf",
      },
    ],
  },
];

function PaperCard({ title, summary, pdf }) {
  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
      className="relative rounded-xl border border-solid border-dark/20 bg-light p-6 dark:border-light/20 dark:bg-dark hover:border-primary dark:hover:border-primaryDark transition-colors duration-300"
    >
      <h4 className="text-lg font-bold mb-2 text-dark dark:text-light">{title}</h4>
      <p className="font-medium text-sm text-dark/60 dark:text-light/60 mb-4">{summary}</p>
      {pdf && (
        <a
          href={`/papers/${pdf}`}
          target="_blank"
          rel="noopener noreferrer"
          download
          className="inline-flex items-center gap-1 text-sm font-semibold text-primary dark:text-primaryDark hover:underline"
        >
          Download PDF &rarr;
        </a>
      )}
    </motion.div>
  );
}

export default function Papers() {
  const sourcePapers = categories.reduce((sum, cat) => sum + cat.papers.length, 0);

  return (
    <>
      <Head>
        <title>Papers | Egoista</title>
        <meta
          name="description"
          content="Complete publication portfolio for the Egoista membrane computing framework. 36 papers, zero free parameters."
        />
      </Head>
      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Publication Portfolio"
            className="mb-8 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          {/* Paper count summary */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-center text-lg font-medium text-dark/60 dark:text-light/60 mb-16 md:text-base"
          >
            {sourcePapers} source papers + 10 vehicle publications = {sourcePapers + 10} total papers available
          </motion.p>

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
                  <PaperCard
                    key={paperIndex}
                    title={paper.title}
                    summary={paper.summary}
                    pdf={paper.pdf}
                  />
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
                {sourcePapers + 10} papers
              </p>
              <p className="text-2xl font-medium text-dark/75 dark:text-light/75 md:text-xl sm:text-lg">
                Zero free parameters
              </p>
              <p className="text-base font-medium text-dark/50 dark:text-light/50 mt-3">
                {sourcePapers} source papers &middot; 10 vehicle publications &middot; All downloadable
              </p>
            </div>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

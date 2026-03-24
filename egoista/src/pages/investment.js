import Layout from "@/components/Layout";
import Head from "next/head";
import { motion, useInView, useMotionValue, useSpring } from "framer-motion";
import { useEffect, useRef } from "react";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import Link from "next/link";

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

const fadeInUp = {
  initial: { y: 50, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

function ProblemCard({ title, description }) {
  return (
    <motion.div
      {...fadeInUp}
      className="relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark"
    >
      <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
      <h3 className="text-2xl font-bold mb-3 text-red-600 dark:text-red-400 md:text-xl">
        {title}
      </h3>
      <p className="font-medium text-light/75">{description}</p>
    </motion.div>
  );
}

function SolutionCard({ title, description }) {
  return (
    <motion.div
      {...fadeInUp}
      className="relative rounded-2xl border-2 border-solid border-gold bg-light p-8 dark:border-gold dark:bg-dark"
    >
      <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-gold/20 dark:bg-gold/10" />
      <h3 className="text-2xl font-bold mb-3 text-primaryDark md:text-xl">
        {title}
      </h3>
      <p className="font-medium text-light/75">{description}</p>
    </motion.div>
  );
}

const comparisonData = [
  { category: "Sensors", conventional: "$100k+ LiDAR, radar, cameras", membrane: "Lipid membrane surface (~$500)" },
  { category: "Computation", conventional: "O(N) forward simulation", membrane: "O(log\u2083 N) backward completion" },
  { category: "Weather", conventional: "Degraded in rain/fog/snow", membrane: "Enhanced -- more information" },
  { category: "Positioning", conventional: "GPS + HD maps (cm accuracy)", membrane: "S-entropy coordinates (topological)" },
  { category: "Safety Model", conventional: "Probabilistic prediction", membrane: "Sufficiency recognition (proven)" },
  { category: "Cost per Vehicle", conventional: "$150,000 - $300,000", membrane: "$15,000 - $30,000" },
];

const roadmap = [
  {
    phase: "Phase 1",
    title: "Computational Validation",
    status: "DONE",
    description:
      "30+ papers published. All 13/13 theoretical validations passed. Complete mathematical framework derived from single axiom. Zero free parameters.",
  },
  {
    phase: "Phase 2",
    title: "Membrane Fabrication & Testing",
    status: "IN PROGRESS",
    description:
      "Lipid bilayer membrane fabrication. Counting loop implementation in biological substrate. Laboratory validation of O(log\u2083 N) navigation.",
  },
  {
    phase: "Phase 3",
    title: "Vehicle Prototype & Licensing",
    status: "UPCOMING",
    description:
      "Integration into vehicle platform. Road testing. Licensing to OEMs. Target: membrane computing module as drop-in replacement for conventional AV stack.",
  },
];

export default function Investment() {
  return (
    <>
      <Head>
        <title>Invest | Egoista</title>
        <meta
          name="description"
          content="Invest in the future of autonomous driving. Membrane computing replaces prediction with mathematical certainty."
        />
      </Head>
      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="The Future of Autonomous Driving"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:!text-6xl xs:!text-4xl sm:mb-8"
          />

          {/* Hero Stat */}
          <motion.div
            {...fadeInUp}
            className="w-full flex flex-col items-center justify-center mb-24 md:mb-16"
          >
            <span className="text-9xl font-bold text-primaryDark md:text-7xl sm:text-5xl">
              $2.3 Trillion
            </span>
            <p className="text-2xl font-medium text-light/75 mt-4 md:text-xl sm:text-lg">
              Autonomous vehicle market by 2030
            </p>
          </motion.div>

          {/* Problem Section */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              The Problem
            </h2>
            <div className="grid grid-cols-3 gap-8 md:grid-cols-1 md:gap-6">
              <ProblemCard
                title="Prediction Fails"
                description="Current AV systems predict the future, but humans don't. Lyapunov divergence renders prediction useless in 0.5-2 seconds. This is not an engineering problem -- it is a mathematical impossibility."
              />
              <ProblemCard
                title="$100k+ Sensors"
                description="Every autonomous vehicle requires LiDAR, radar, and multiple camera arrays costing over $100,000 per vehicle. This makes mass-market autonomous vehicles economically impossible."
              />
              <ProblemCard
                title="Weather Blindness"
                description="Conventional sensors degrade catastrophically in rain, fog, and snow. LiDAR scatters, cameras blur, radar loses resolution. The vehicles that need autonomy most cannot have it."
              />
            </div>
          </motion.div>

          {/* Solution Section */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              The Solution
            </h2>
            <div className="grid grid-cols-3 gap-8 md:grid-cols-1 md:gap-6">
              <SolutionCard
                title="Membrane Computing"
                description="Biological lipid membranes perform 10^28 operations per second on their surface. We harness this as a computational substrate -- the membrane IS the computer."
              />
              <SolutionCard
                title="Zero Prediction"
                description="Sufficiency recognition replaces forward simulation. The vehicle navigates backward through partition space at O(log_3 N), recognizing when conditions are sufficient to act."
              />
              <SolutionCard
                title="Weather Enhanced"
                description="Bad weather means more molecular interactions on the membrane surface. Rain, fog, and snow provide MORE information, not less. The harder the conditions, the better the system performs."
              />
            </div>
          </motion.div>

          {/* Key Metrics */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              Key Metrics
            </h2>
            <div className="grid grid-cols-4 gap-8 md:grid-cols-2 sm:grid-cols-1 sm:gap-6">
              <div className="flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-solid border-dark dark:border-light">
                <span className="text-6xl font-bold text-primaryDark md:text-5xl sm:text-4xl">
                  <AnimatedNumberFramerMotion value={30} suffix="+" />
                </span>
                <h3 className="text-xl font-medium text-light/75 mt-4 text-center md:text-lg">
                  Papers (IP Portfolio)
                </h3>
              </div>
              <div className="flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-solid border-dark dark:border-light">
                <span className="text-6xl font-bold text-primaryDark md:text-5xl sm:text-4xl">
                  <AnimatedNumberFramerMotion value={13} />/13
                </span>
                <h3 className="text-xl font-medium text-light/75 mt-4 text-center md:text-lg">
                  Validations Passed
                </h3>
              </div>
              <div className="flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-solid border-dark dark:border-light">
                <span className="text-6xl font-bold text-primaryDark md:text-5xl sm:text-4xl">
                  10<sup>28</sup>
                </span>
                <h3 className="text-xl font-medium text-light/75 mt-4 text-center md:text-lg">
                  Ops/s Processing
                </h3>
              </div>
              <div className="flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-solid border-dark dark:border-light">
                <span className="text-6xl font-bold text-primaryDark md:text-5xl sm:text-4xl">
                  <AnimatedNumberFramerMotion value={0} />
                </span>
                <h3 className="text-xl font-medium text-light/75 mt-4 text-center md:text-lg">
                  Free Parameters
                </h3>
              </div>
            </div>
          </motion.div>

          {/* Comparison Table */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              Conventional vs. Membrane AV
            </h2>
            <div className="w-full overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="text-left p-4 border-b-2 border-dark dark:border-light font-bold text-lg">
                      Category
                    </th>
                    <th className="text-left p-4 border-b-2 border-dark dark:border-light font-bold text-lg text-red-600 dark:text-red-400">
                      Conventional AV
                    </th>
                    <th className="text-left p-4 border-b-2 border-dark dark:border-light font-bold text-lg text-primaryDark">
                      Membrane AV
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonData.map((row, index) => (
                    <tr
                      key={index}
                      className={index % 2 === 0 ? "bg-dark/5 dark:bg-light/5" : ""}
                    >
                      <td className="p-4 font-bold border-b border-dark/20 dark:border-light/20">
                        {row.category}
                      </td>
                      <td className="p-4 font-medium border-b border-dark/20 dark:border-light/20 text-light/75">
                        {row.conventional}
                      </td>
                      <td className="p-4 font-medium border-b border-dark/20 dark:border-light/20 text-primaryDark">
                        {row.membrane}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>

          {/* Roadmap */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="font-bold text-6xl mb-12 w-full text-center md:text-4xl xs:text-3xl">
              Roadmap
            </h2>
            <div className="grid grid-cols-3 gap-8 md:grid-cols-1 md:gap-6">
              {roadmap.map((item, index) => (
                <motion.div
                  key={index}
                  {...fadeInUp}
                  transition={{ duration: 0.6, delay: index * 0.15 }}
                  className="relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark"
                >
                  <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-lg font-bold text-dark/50 dark:text-light/50">
                      {item.phase}
                    </span>
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-bold ${
                        item.status === "DONE"
                          ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                          : item.status === "IN PROGRESS"
                          ? "bg-gold/20 text-gold dark:bg-gold/30 dark:text-gold"
                          : "bg-dark/10 text-dark/50 dark:bg-light/10 dark:text-light/50"
                      }`}
                    >
                      {item.status}
                    </span>
                  </div>
                  <h3 className="text-2xl font-bold mb-3 text-primaryDark md:text-xl">
                    {item.title}
                  </h3>
                  <p className="font-medium text-light/75">
                    {item.description}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Molecular Navigation */}
          <motion.div {...fadeInUp} className="w-full mb-16">
            <h2 className="font-bold text-6xl mb-4 w-full text-center md:text-4xl xs:text-3xl text-light">
              Molecular Navigation
            </h2>
            <p className="text-lg text-light/70 text-center mb-8 max-w-3xl mx-auto">
              The atmosphere remembers. Exhaust trails persist for hours, encoding optimal paths
              discovered by millions of drivers. The membrane reads this molecular memory directly.
            </p>
            <div className="grid grid-cols-4 gap-4 md:grid-cols-2 sm:grid-cols-1">
              {[
                { title: "Night Vision", stat: "100m", desc: "Detection range in total darkness" },
                { title: "Brake Warning", stat: "240ms", desc: "Advance warning before brake lights" },
                { title: "Around Corner", stat: "10-20s", desc: "Hidden vehicle detection via exhaust diffusion" },
                { title: "Optimal Path", stat: "0.15m", desc: "RMS error extracting path from 500 drivers" },
                { title: "Convoy", stat: "92%", desc: "Variance reduction in spacing (no V2V)" },
                { title: "Hazard Gaps", stat: "100%", desc: "Pothole detection from trail absence" },
                { title: "V2A2V", stat: "\u221E", desc: "Vehicle-to-Atmosphere-to-Vehicle bandwidth" },
                { title: "5/5", stat: "\u2713", desc: "All navigation experiments validated" },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.05 }}
                  className="p-4 border border-light/10 rounded-xl text-center"
                >
                  <div className="text-2xl font-bold text-primaryDark">{item.stat}</div>
                  <div className="text-sm font-bold text-gold mt-1">{item.title}</div>
                  <div className="text-xs text-light/50 mt-1">{item.desc}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Contact CTA */}
          <motion.div
            {...fadeInUp}
            className="w-full flex flex-col items-center justify-center mb-16"
          >
            <h2 className="font-bold text-6xl mb-8 w-full text-center md:text-4xl xs:text-3xl">
              Join the Revolution
            </h2>
            <p className="text-xl font-medium text-light/75 mb-8 text-center max-w-2xl md:text-lg">
              We are seeking strategic partners and investors who understand that the future of
              autonomous driving is not better prediction -- it is no prediction at all.
            </p>
            <Link
              href="mailto:kundai.sachikonye@wzw.tum.de"
              className="flex items-center rounded-lg border-2 border-gold bg-gold/10 p-6 px-12 text-lg font-semibold capitalize text-gold hover:bg-gold hover:text-dark md:p-4 md:px-8 md:text-base transition-all duration-300"
            >
              Get in Touch
            </Link>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

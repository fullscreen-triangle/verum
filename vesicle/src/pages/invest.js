import Layout from "@/components/Layout";
import Head from "next/head";
import { motion, useInView, useMotionValue, useSpring } from "framer-motion";
import { useEffect, useRef } from "react";
import TransitionEffect from "@/components/TransitionEffect";
import Link from "next/link";

function AnimatedNumber({ value, suffix = "" }) {
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

const problemCards = [
  {
    title: "Prediction Fails",
    description: "Current AV systems predict the future, but Lyapunov divergence renders prediction useless in 0.5\u20132 seconds. This is not an engineering problem \u2014 it is a mathematical impossibility.",
  },
  {
    title: "Sensors Cost $100k+",
    description: "Every autonomous vehicle requires LiDAR, radar, and multiple camera arrays costing over $100,000 per vehicle. This makes mass-market autonomous vehicles economically impossible.",
  },
  {
    title: "Weather Blindness",
    description: "Conventional sensors degrade catastrophically in rain, fog, and snow. LiDAR scatters, cameras blur, radar loses resolution. The vehicles that need autonomy most cannot have it.",
  },
];

const solutionCards = [
  {
    title: "Membrane Computing",
    stat: "10\u00B2\u2078 ops/s",
    description: "Biological lipid membranes perform 10\u00B2\u2078 operations per second on their surface. The membrane IS the computer \u2014 every lipid oscillation is a computation.",
  },
  {
    title: "Zero Prediction",
    stat: "O(log\u2083 N)",
    description: "Backward trajectory completion replaces forward simulation. The vehicle navigates through partition space, recognizing when conditions are sufficient to act. Chaos-free by construction.",
  },
  {
    title: "Weather Enhanced",
    stat: "+\u221E in rain",
    description: "Bad weather means more molecular interactions on the membrane surface. Rain, fog, and snow provide MORE information, not less. The harder the conditions, the better the system performs.",
  },
];

const platformLayers = [
  { num: 1, name: "Lipid Membrane Sensor Array", stat: "10\u00B2\u2078 ops/s", color: "#2AA198" },
  { num: 2, name: "Buhera Operating System", stat: "Categorical OS", color: "#C6A962" },
  { num: 3, name: "Trajectory Completion Computing", stat: "O(log\u2083 N)", color: "#58E6D9" },
  { num: 4, name: "Cynegeticus Positioning", stat: "GPS-free", color: "#D4AF37" },
  { num: 5, name: "Ober Atmospheric Scripting", stat: "Weather = data", color: "#2AA198" },
  { num: 6, name: "Sango-Rine-Shumba Protocol", stat: "V2A2V", color: "#C6A962" },
  { num: 7, name: "Zangalewa AI Interceptor", stat: "AI bridge", color: "#58E6D9" },
  { num: 8, name: "Federated Multi-Modal Understanding", stat: "Privacy-first", color: "#D4AF37" },
  { num: 9, name: "Current Flux Mechanism", stat: "Signal out", color: "#2AA198" },
];

const navMetrics = [
  { title: "Night Vision", stat: "100m", desc: "Detection range in total darkness" },
  { title: "Brake Warning", stat: "240ms", desc: "Advance warning before brake lights" },
  { title: "Around Corner", stat: "10-20s", desc: "Hidden vehicle detection via exhaust diffusion" },
  { title: "Optimal Path", stat: "0.15m", desc: "RMS error extracting path from 500 drivers" },
  { title: "Convoy", stat: "92%", desc: "Variance reduction in spacing (no V2V)" },
  { title: "Hazard Gaps", stat: "100%", desc: "Pothole detection from trail absence" },
  { title: "V2A2V", stat: "\u221E", desc: "Vehicle-to-Atmosphere-to-Vehicle bandwidth" },
  { title: "Validated", stat: "5/5", desc: "All navigation experiments validated" },
];

const marketData = [
  { label: "TAM", value: "$2.3T", desc: "Global autonomous vehicle market" },
  { label: "SAM", value: "$300B", desc: "Premium + trucking + defense" },
  { label: "SOM", value: "$10B", desc: "First 5 years addressable" },
];

const roadmap = [
  {
    phase: "Phase 1", title: "Theory + Validation", status: "DONE",
    description: "30+ papers published. All 18/18 theoretical validations passed. Complete mathematical framework derived from single axiom. Zero free parameters. Full 9-layer stack defined.",
  },
  {
    phase: "Phase 2", title: "Prototype + Funding", status: "IN PROGRESS",
    description: "Lipid bilayer membrane fabrication. Counting loop implementation in biological substrate. Laboratory validation of O(log\u2083 N) navigation. Seed round targeting.",
  },
  {
    phase: "Phase 3", title: "Vehicle Integration + Deployment", status: "UPCOMING",
    description: "Integration into vehicle platform. Road testing across weather conditions. OEM licensing. Target: membrane computing module as drop-in replacement for conventional AV stack.",
  },
];

export default function Invest() {
  return (
    <>
      <Head>
        <title>Invest | Vesicle</title>
        <meta
          name="description"
          content="Invest in the future of autonomous driving. Membrane computing replaces prediction with mathematical certainty. $2.3 trillion market."
        />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        <Layout className="pt-16">

          {/* Hero */}
          <motion.div
            {...fadeInUp}
            className="text-center mb-24 md:mb-16"
          >
            <h1 className="text-6xl font-bold mb-6 md:text-4xl sm:text-3xl">
              The Future of Autonomous Driving
            </h1>
            <div className="mt-8">
              <span className="text-9xl font-bold text-gold md:text-7xl sm:text-5xl">
                $2.3 Trillion
              </span>
              <p className="text-2xl text-light/60 mt-4 md:text-xl sm:text-lg">
                Autonomous vehicle market by 2030
              </p>
            </div>
          </motion.div>

          {/* The Problem */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="text-5xl font-bold mb-12 text-center md:text-4xl">
              The Problem
            </h2>
            <div className="grid grid-cols-3 gap-6 md:grid-cols-1">
              {problemCards.map((card, i) => (
                <motion.div
                  key={card.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="p-6 rounded-xl border border-red-500/30 bg-red-500/5"
                >
                  <h3 className="text-xl font-bold mb-3 text-red-400">{card.title}</h3>
                  <p className="text-light/60 text-sm leading-relaxed">{card.description}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* The Solution */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="text-5xl font-bold mb-12 text-center md:text-4xl">
              The Solution
            </h2>
            <div className="grid grid-cols-3 gap-6 md:grid-cols-1">
              {solutionCards.map((card, i) => (
                <motion.div
                  key={card.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="p-6 rounded-xl border border-gold/30 bg-gold/5"
                >
                  <div className="text-2xl font-bold text-gold mb-2">{card.stat}</div>
                  <h3 className="text-xl font-bold mb-3 text-primaryDark">{card.title}</h3>
                  <p className="text-light/60 text-sm leading-relaxed">{card.description}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* The Platform */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="text-5xl font-bold mb-12 text-center md:text-4xl">
              The Platform
            </h2>
            <div className="max-w-3xl mx-auto space-y-2">
              {platformLayers.map((layer, i) => (
                <motion.div
                  key={layer.num}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.04 }}
                  className="flex items-center gap-4 p-3 border border-light/10 rounded-lg hover:border-light/20 transition-all"
                >
                  <span
                    className="text-xs font-mono font-bold w-6 text-center"
                    style={{ color: layer.color }}
                  >
                    {layer.num}
                  </span>
                  <span className="flex-1 font-medium text-sm">{layer.name}</span>
                  <span className="text-xs text-light/40 font-mono">{layer.stat}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Key Metrics */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="text-5xl font-bold mb-12 text-center md:text-4xl">
              Key Metrics
            </h2>
            <div className="grid grid-cols-4 gap-6 md:grid-cols-2 sm:grid-cols-1">
              <div className="flex flex-col items-center justify-center p-8 rounded-xl border border-light/10">
                <span className="text-5xl font-bold text-gold md:text-4xl">
                  <AnimatedNumber value={30} suffix="+" />
                </span>
                <span className="text-sm text-light/50 mt-3 text-center">Papers (IP Portfolio)</span>
              </div>
              <div className="flex flex-col items-center justify-center p-8 rounded-xl border border-light/10">
                <span className="text-5xl font-bold text-gold md:text-4xl">
                  <AnimatedNumber value={18} />/18
                </span>
                <span className="text-sm text-light/50 mt-3 text-center">Validations Passed</span>
              </div>
              <div className="flex flex-col items-center justify-center p-8 rounded-xl border border-light/10">
                <span className="text-5xl font-bold text-gold md:text-4xl">
                  10<sup>28</sup>
                </span>
                <span className="text-sm text-light/50 mt-3 text-center">Ops/s Processing</span>
              </div>
              <div className="flex flex-col items-center justify-center p-8 rounded-xl border border-light/10">
                <span className="text-5xl font-bold text-gold md:text-4xl">
                  <AnimatedNumber value={0} />
                </span>
                <span className="text-sm text-light/50 mt-3 text-center">Free Parameters</span>
              </div>
            </div>
          </motion.div>

          {/* Molecular Navigation */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="text-5xl font-bold mb-4 text-center md:text-4xl">
              Molecular Navigation
            </h2>
            <p className="text-lg text-light/50 text-center mb-10 max-w-3xl mx-auto">
              The atmosphere remembers. Exhaust trails persist for hours, encoding optimal paths
              discovered by millions of drivers. The membrane reads this molecular memory directly.
            </p>
            <div className="grid grid-cols-4 gap-4 md:grid-cols-2 sm:grid-cols-1">
              {navMetrics.map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.05 }}
                  className="p-4 border border-light/10 rounded-xl text-center hover:border-light/20 transition-all"
                >
                  <div className="text-2xl font-bold text-gold">{item.stat}</div>
                  <div className="text-sm font-bold text-primaryDark mt-1">{item.title}</div>
                  <div className="text-xs text-light/40 mt-1">{item.desc}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Market */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="text-5xl font-bold mb-12 text-center md:text-4xl">
              Market Opportunity
            </h2>
            <div className="grid grid-cols-3 gap-6 md:grid-cols-1">
              {marketData.map((m, i) => (
                <motion.div
                  key={m.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="text-center p-8 border border-light/10 rounded-xl"
                >
                  <div className="text-xs font-mono text-light/40 uppercase tracking-widest mb-2">{m.label}</div>
                  <div className="text-5xl font-bold text-gold md:text-4xl">{m.value}</div>
                  <div className="text-sm text-light/50 mt-2">{m.desc}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Roadmap */}
          <motion.div {...fadeInUp} className="mb-24 md:mb-16">
            <h2 className="text-5xl font-bold mb-12 text-center md:text-4xl">
              Roadmap
            </h2>
            <div className="grid grid-cols-3 gap-6 md:grid-cols-1">
              {roadmap.map((item, i) => (
                <motion.div
                  key={item.phase}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="p-6 rounded-xl border border-light/10"
                >
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-sm font-bold text-light/40">{item.phase}</span>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-bold ${
                        item.status === "DONE"
                          ? "bg-green-900/50 text-green-400 border border-green-500/30"
                          : item.status === "IN PROGRESS"
                          ? "bg-gold/10 text-gold border border-gold/30"
                          : "bg-light/5 text-light/40 border border-light/10"
                      }`}
                    >
                      {item.status}
                    </span>
                  </div>
                  <h3 className="text-xl font-bold mb-3 text-primaryDark">{item.title}</h3>
                  <p className="text-light/60 text-sm leading-relaxed">{item.description}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* CTA */}
          <motion.div
            {...fadeInUp}
            className="text-center mb-16"
          >
            <h2 className="text-5xl font-bold mb-6 md:text-4xl">
              Join the Revolution
            </h2>
            <p className="text-xl text-light/50 mb-10 max-w-2xl mx-auto md:text-lg">
              We are seeking strategic partners and investors who understand that the future of
              autonomous driving is not better prediction — it is no prediction at all.
            </p>
            <Link
              href="mailto:kundai.sachikonye@wzw.tum.de"
              className="inline-block px-12 py-4 border-2 border-gold bg-gold/10 text-gold text-lg font-semibold rounded-lg hover:bg-gold hover:text-dark transition-all duration-300"
            >
              Get in Touch
            </Link>
          </motion.div>

        </Layout>
      </main>
    </>
  );
}

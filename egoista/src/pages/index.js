import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import Link from "next/link";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

const LamborghiniScene = dynamic(
  () => import("@/components/LamborghiniModel"),
  { ssr: false }
);

const stackPreview = [
  { name: "Membrane", desc: "10\u00B2\u2078 ops/s computing surface", color: "#2AA198" },
  { name: "Buhera OS", desc: "Categorical operating system", color: "#C6A962" },
  { name: "TCC", desc: "O(log\u2083 N) backward navigation", color: "#58E6D9" },
  { name: "Cynegeticus", desc: "GPS-free positioning", color: "#D4AF37" },
  { name: "Ober", desc: "Weather as information", color: "#2AA198" },
  { name: "Sango", desc: "Gas-based distributed internet", color: "#C6A962" },
  { name: "Zangalewa", desc: "AI interceptor layer", color: "#58E6D9" },
  { name: "Federated", desc: "Privacy-preserving learning", color: "#D4AF37" },
  { name: "Current Flux", desc: "Membrane signal transduction", color: "#2AA198" },
];

const stats = [
  { value: "30+", label: "Papers" },
  { value: "10\u00B2\u2078", label: "ops/s" },
  { value: "18/18", label: "Validations" },
  { value: "0", label: "Free Parameters" },
];

export default function Home() {
  return (
    <>
      <Head>
        <title>Vesicle | Membrane Computing Autonomous Vehicles</title>
        <meta name="description" content="The complete autonomous vehicle platform. 9-layer stack from lipid membrane hardware to federated AI." />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        {/* Hero */}
        <section className="relative h-screen flex items-center justify-center overflow-hidden">
          <div className="absolute inset-0 hidden md:block">
            <LamborghiniScene />
          </div>
          <div className="absolute inset-0 md:hidden bg-gradient-radial from-dark via-dark to-dark/80" />
          <div className="relative z-10 text-center pointer-events-none px-4">
            <motion.h1
              initial={{ opacity: 0, y: -30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-8xl font-bold tracking-widest mb-4 md:text-6xl sm:text-5xl"
              style={{ textShadow: "0 0 40px rgba(198,169,98,0.3)" }}
            >
              VESICLE
            </motion.h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="text-xl text-light/70 mb-2 md:text-lg"
            >
              The Complete Autonomous Vehicle Platform
            </motion.p>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.6 }}
              className="text-sm text-light/40 mb-8 tracking-wider uppercase"
            >
              9 layers &middot; 30+ papers &middot; Zero free parameters
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.5 }}
              className="flex gap-4 justify-center pointer-events-auto"
            >
              <Link href="/platform" className="px-8 py-3 border border-gold text-gold hover:bg-gold hover:text-dark transition-all duration-300 rounded-lg font-semibold">
                Explore Platform
              </Link>
              <Link href="/invest" className="px-8 py-3 bg-gold text-dark hover:bg-gold/80 transition-all duration-300 rounded-lg font-semibold">
                Invest
              </Link>
            </motion.div>
          </div>
        </section>

        {/* Platform Preview */}
        <section className="px-8 py-20 max-w-7xl mx-auto md:px-4">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-4xl font-bold text-center mb-4"
          >
            One Platform. Nine Layers.
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-light/50 text-center mb-12 max-w-2xl mx-auto"
          >
            From lipid membrane hardware to federated AI — a complete autonomous vehicle stack derived from a single axiom.
          </motion.p>
          <div className="grid grid-cols-3 gap-3 md:grid-cols-2 sm:grid-cols-1">
            {stackPreview.map((s, i) => (
              <motion.div
                key={s.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
              >
                <Link href="/platform" className="block p-4 border border-light/10 rounded-lg hover:border-light/30 transition-all group">
                  <div className="text-xs font-mono tracking-wider mb-1" style={{ color: s.color }}>
                    LAYER {i + 1}
                  </div>
                  <div className="font-bold text-light group-hover:text-gold transition-colors">{s.name}</div>
                  <div className="text-sm text-light/50">{s.desc}</div>
                </Link>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Stats */}
        <section className="px-8 py-16 border-t border-b border-light/10">
          <div className="max-w-4xl mx-auto grid grid-cols-4 gap-8 md:grid-cols-2">
            {stats.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl font-bold text-gold">{s.value}</div>
                <div className="text-sm text-light/50 uppercase tracking-wider mt-1">{s.label}</div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section className="px-8 py-20 text-center">
          <h2 className="text-3xl font-bold mb-4">Built on 30+ Papers of Hard Physics</h2>
          <p className="text-light/50 mb-8 max-w-xl mx-auto">
            Every claim derived from first principles. Every parameter predicted, not fitted. Every validation passed.
          </p>
          <div className="flex gap-4 justify-center">
            <Link href="/papers" className="px-6 py-3 border border-light/20 text-light/70 hover:border-gold hover:text-gold transition-all rounded-lg">
              Read the Papers
            </Link>
            <Link href="/membrane" className="px-6 py-3 border border-light/20 text-light/70 hover:border-membrane hover:text-membrane transition-all rounded-lg">
              See the Membrane
            </Link>
          </div>
        </section>
      </main>
    </>
  );
}

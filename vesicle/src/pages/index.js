import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import Link from "next/link";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

const HeroScene = dynamic(() => import("@/components/HeroScene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-dark">
      <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-[#2AA198] border-t-transparent" />
    </div>
  ),
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Vesicle | Membrane Computing Autonomous Vehicles</title>
        <meta name="description" content="Membrane computing for autonomous vehicles." />
      </Head>
      <TransitionEffect />
      <main className="w-full bg-dark text-light">
        {/* Hero - full viewport */}
        <section className="relative h-screen">
          <div className="absolute inset-0">
            <HeroScene />
          </div>

          {/* VESICLE text overlay */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10">
            <h1
              style={{
                fontSize: "clamp(4rem, 12vw, 10rem)",
                fontWeight: 900,
                letterSpacing: "0.15em",
                color: "transparent",
                WebkitTextStroke: "1.5px rgba(255,255,255,0.25)",
                textShadow: "0 0 60px rgba(42,161,152,0.15)",
              }}
            >
              VESICLE
            </h1>
          </div>

          {/* Scroll hint */}
          <motion.div
            className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10"
            animate={{ y: [0, 8, 0] }}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            <span className="text-xs text-light/30 tracking-widest uppercase">
              Scroll to explore
            </span>
          </motion.div>
        </section>

        {/* Below fold */}
        <section className="h-screen flex items-center justify-center px-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <p className="text-2xl text-light/50 mb-8 tracking-wide">
              Membrane Computing for Autonomous Vehicles
            </p>
            <div className="flex gap-4 justify-center">
              <Link
                href="/platform"
                className="px-8 py-3 border border-gold text-gold hover:bg-gold hover:text-dark transition-all rounded-lg"
              >
                Explore Platform
              </Link>
              <Link
                href="/invest"
                className="px-8 py-3 bg-gold/10 border border-gold/30 text-gold hover:bg-gold hover:text-dark transition-all rounded-lg"
              >
                Invest
              </Link>
            </div>
          </motion.div>
        </section>
      </main>
    </>
  );
}

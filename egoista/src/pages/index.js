import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import AnimatedText from "@/components/AnimatedText";
import Link from "next/link";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

/* Load the 3D scene only on the client (no SSR for WebGL) */
const LamborghiniScene = dynamic(
  () => import("@/components/LamborghiniModel"),
  { ssr: false }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>Egoista | Membrane Computing for Autonomous Vehicles</title>
        <meta
          name="description"
          content="Egoista: Autonomous driving through membrane computing. Oscillatory dynamics, phase-locked ensembles, and S-entropy governance on a Lamborghini-inspired platform."
        />
      </Head>

      <TransitionEffect />

      <main className="flex items-center text-dark w-full min-h-screen dark:text-light relative">
        {/* ---------------------------------------------------------------- */}
        {/*  Full-screen 3D model (hidden on small screens)                  */}
        {/* ---------------------------------------------------------------- */}
        <div className="absolute inset-0 z-0 hidden sm:hidden md:block">
          <LamborghiniScene />
        </div>

        {/* Static gradient fallback for mobile */}
        <div
          className="absolute inset-0 z-0 block md:hidden"
          style={{
            background:
              "radial-gradient(ellipse at 60% 50%, #1a1a2e 0%, #0d0d0d 70%)",
          }}
        />

        {/* ---------------------------------------------------------------- */}
        {/*  Overlaid content                                                */}
        {/* ---------------------------------------------------------------- */}
        <Layout className="relative z-10 !bg-transparent">
          <div className="pointer-events-none flex w-full flex-col items-center justify-center">
            {/* Large heading */}
            <motion.h1
              className="text-8xl font-black tracking-tighter text-dark dark:text-light
                xl:text-7xl lg:text-6xl md:text-5xl sm:text-4xl"
              style={{
                textShadow:
                  "0 2px 30px rgba(0,0,0,0.6), 0 0px 80px rgba(198,169,98,0.25)",
              }}
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, ease: "easeOut" }}
            >
              EGOISTA
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              className="mt-4 max-w-2xl text-center text-xl font-medium text-dark/80
                dark:text-light/80 md:text-lg sm:text-base"
              style={{
                textShadow: "0 1px 12px rgba(0,0,0,0.5)",
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
            >
              Autonomous driving through membrane computing
            </motion.p>

            {/* Decorative divider */}
            <motion.div
              className="my-8 h-px w-48 bg-gradient-to-r from-transparent via-[#C6A962] to-transparent"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 1.2, delay: 0.6, ease: "easeOut" }}
            />

            {/* CTA buttons */}
            <motion.div
              className="pointer-events-auto mt-2 flex items-center gap-6 sm:flex-col sm:gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.8, ease: "easeOut" }}
            >
              <Link
                href="/framework"
                className="rounded-lg border-2 border-[#C6A962] bg-[#C6A962]/10 px-8 py-3
                  text-lg font-semibold capitalize text-dark backdrop-blur-sm
                  transition-all duration-300
                  hover:bg-[#C6A962] hover:text-dark hover:shadow-lg hover:shadow-[#C6A962]/20
                  dark:text-light dark:hover:text-dark
                  md:px-6 md:py-2.5 md:text-base"
              >
                Explore Framework
              </Link>

              <Link
                href="/investment"
                className="rounded-lg border-2 border-[#C6A962] bg-[#C6A962] px-8 py-3
                  text-lg font-semibold capitalize text-dark
                  transition-all duration-300
                  hover:bg-transparent hover:text-dark hover:shadow-lg hover:shadow-[#C6A962]/20
                  dark:hover:text-light
                  md:px-6 md:py-2.5 md:text-base"
              >
                Invest
              </Link>
            </motion.div>

            {/* Tertiary link */}
            <motion.div
              className="pointer-events-auto mt-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 1.2 }}
            >
              <Link
                href="/membrane"
                className="text-sm font-medium uppercase tracking-widest text-dark/50
                  underline underline-offset-4 transition-colors hover:text-[#C6A962]
                  dark:text-light/50 dark:hover:text-[#C6A962]"
                style={{ textShadow: "0 1px 8px rgba(0,0,0,0.4)" }}
              >
                View Membrane Shader
              </Link>
            </motion.div>
          </div>
        </Layout>
      </main>
    </>
  );
}

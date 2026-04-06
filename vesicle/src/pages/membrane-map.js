import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const MembraneMapScene = dynamic(
  () => import("@/components/MembraneMapScene"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[70vh] w-full items-center justify-center bg-dark">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-[#2AA198] border-t-transparent" />
      </div>
    ),
  }
);

const fadeUp = {
  initial: { y: 40, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

export default function MembraneMap() {
  return (
    <>
      <Head>
        <title>Membrane Map | Vesicle</title>
        <meta
          name="description"
          content="Membrane shader applied to a 3D car on a Munich map -- visualizing lipid oscillation in geographic context."
        />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        {/* Map Section */}
        <section className="w-full h-[70vh] relative">
          <MembraneMapScene />
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-dark to-transparent h-24 pointer-events-none" />
          <div className="absolute top-4 left-4 bg-dark/80 backdrop-blur-sm rounded-lg p-3 border border-light/10 z-10">
            <div className="text-xs text-gold uppercase tracking-wider font-bold">
              Munich -- Membrane Shader
            </div>
            <div className="text-xs text-light/40 mt-1">
              Lipid oscillation on geographic surface
            </div>
          </div>
        </section>

        <Layout>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-5xl mx-auto"
          >
            <h1 className="text-5xl font-bold text-center mb-4 md:text-4xl">
              Membrane in Context
            </h1>
            <p className="text-light/50 text-center mb-12 text-lg max-w-2xl mx-auto">
              The membrane shader visualized on a vehicle within the Munich
              urban environment. Teal and gold pulses represent lipid
              oscillation states coupling to atmospheric O&#x2082; ensembles.
            </p>

            <div className="grid grid-cols-2 gap-6 mb-16 md:grid-cols-1">
              <motion.div
                {...fadeUp}
                className="p-6 border border-light/10 rounded-2xl"
              >
                <h3 className="text-lg font-bold mb-3 text-primaryDark">
                  Geographic Coupling
                </h3>
                <p className="text-sm text-light/60 leading-relaxed">
                  The membrane&apos;s sensing capability varies with geographic
                  context. Urban canyons create molecular channeling effects that
                  amplify exhaust trail persistence. Munich&apos;s street grid
                  near the Hochbunker creates natural waveguides for molecular
                  information.
                </p>
              </motion.div>
              <motion.div
                {...fadeUp}
                transition={{ delay: 0.1 }}
                className="p-6 border border-light/10 rounded-2xl"
              >
                <h3 className="text-lg font-bold mb-3 text-primaryDark">
                  Shader Representation
                </h3>
                <p className="text-sm text-light/60 leading-relaxed">
                  The pulsing teal-gold pattern represents the lipid bilayer
                  oscillating at its characteristic frequency. Wave patterns
                  encode the phase-locked state of O&#x2082; ensembles. Bright
                  regions indicate high coupling strength with atmospheric
                  molecular information.
                </p>
              </motion.div>
            </div>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

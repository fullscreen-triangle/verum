import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const AirflowScene = dynamic(() => import("@/components/AirflowScene"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[60vh] bg-light/5 rounded-2xl animate-pulse flex items-center justify-center text-light/30">
      Loading airflow model...
    </div>
  ),
});

const fadeUp = {
  initial: { y: 40, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

const concepts = [
  {
    title: "Boundary Layer Reading",
    desc: "The membrane reads the airflow boundary layer -- the thin region where air velocity transitions from zero at the car surface to free-stream. This layer encodes velocity gradients, pressure distribution, and molecular composition in a continuous field the membrane samples at 10^11 Hz.",
    color: "#2AA198",
  },
  {
    title: "Streamline Signatures",
    desc: "Every vehicle creates a unique aerodynamic wake signature. Streamlines encode speed, direction, mass, and even engine state. The membrane detects these wake signatures from vehicles ahead, providing information about traffic conditions beyond line of sight.",
    color: "#D4AF37",
  },
  {
    title: "Turbulent Diffusion",
    desc: "Exhaust plumes spread via turbulent diffusion with D approximately 1 m^2/s. This means molecular trails persist for minutes to hours, encoding the history of every vehicle that passed. The membrane reads these persistent trails as a molecular memory of road usage.",
    color: "#C6A962",
  },
  {
    title: "Pressure Wave Detection",
    desc: "Vehicles generate pressure waves that propagate at the speed of sound. The membrane detects these compression waves, providing 240ms advance warning of braking events -- 4.5 to 8.7 metres at highway speed before brake lights even illuminate.",
    color: "#58E6D9",
  },
];

export default function Airflow() {
  return (
    <>
      <Head>
        <title>Airflow | Vesicle</title>
        <meta
          name="description"
          content="Airflow boundary layer visualization -- how the membrane reads aerodynamic signatures for superhuman perception."
        />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        <Layout className="pt-16">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-5xl mx-auto"
          >
            <h1 className="text-5xl font-bold text-center mb-4 md:text-4xl">
              Airflow Visualization
            </h1>
            <p className="text-light/50 text-center mb-12 text-lg max-w-2xl mx-auto">
              The aerodynamic boundary layer is not just drag -- it is the
              membrane&apos;s primary information channel. Every streamline
              encodes environmental state.
            </p>

            {/* 3D Streamlines */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="mb-16"
            >
              <AirflowScene
                modelPath="/model/airshaper_demo_beta__3d_streamlines.glb"
                height="60vh"
              />
              <p className="text-xs text-light/30 text-center mt-3">
                3D streamline visualization -- drag to rotate, scroll to zoom
              </p>
            </motion.div>

            {/* 2D Streamlines */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="mb-16"
            >
              <h2 className="text-2xl font-bold text-center mb-6">
                2D Streamline Cross-Section
              </h2>
              <AirflowScene
                modelPath="/model/airshaper_demo_beta__2d_streamlines.glb"
                height="50vh"
              />
              <p className="text-xs text-light/30 text-center mt-3">
                Cross-sectional view of airflow patterns around the vehicle body
              </p>
            </motion.div>

            {/* Concepts */}
            <section className="mb-16">
              <h2 className="text-3xl font-bold text-center mb-8">
                How the Membrane Reads the Boundary Layer
              </h2>
              <div className="grid grid-cols-2 gap-6 md:grid-cols-1">
                {concepts.map((item, i) => (
                  <motion.div
                    key={item.title}
                    {...fadeUp}
                    transition={{ delay: i * 0.1 }}
                    className="p-6 border border-light/10 rounded-2xl"
                  >
                    <h3
                      className="text-lg font-bold mb-3"
                      style={{ color: item.color }}
                    >
                      {item.title}
                    </h3>
                    <p className="text-sm text-light/60 leading-relaxed">
                      {item.desc}
                    </p>
                  </motion.div>
                ))}
              </div>
            </section>

            {/* Key insight */}
            <motion.div
              {...fadeUp}
              className="w-full p-8 rounded-2xl border-2 border-primaryDark bg-primaryDark/5 text-center mb-16"
            >
              <h3 className="text-2xl font-bold mb-4 text-primaryDark">
                The Wake as Communication
              </h3>
              <p className="text-lg text-light/80 max-w-2xl mx-auto">
                Every vehicle&apos;s aerodynamic wake is a broadcast signal. The
                membrane transforms traffic from isolated agents into a
                molecular network where every vehicle communicates its state
                through the air itself.
              </p>
            </motion.div>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

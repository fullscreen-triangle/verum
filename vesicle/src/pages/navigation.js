import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const NavigationMap = dynamic(() => import("@/components/MunichMapScene"), { ssr: false });

const fadeIn = {
  initial: { y: 30, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.5 },
};

const experiments = [
  { name: "Night Navigation", result: "100m", desc: "Detection range in total darkness via thermal + molecular perturbations", status: "PASS" },
  { name: "Brake Anticipation", result: "240ms", desc: "Advance warning before brake lights via exhaust composition shift", status: "PASS" },
  { name: "Around Corner", result: "10-20s", desc: "Hidden vehicle detection via exhaust plume diffusion", status: "PASS" },
  { name: "Optimal Path", result: "0.15m RMS", desc: "Path extraction from 500 drivers' cumulative exhaust trails", status: "PASS" },
  { name: "Convoy Formation", result: "92%", desc: "Spacing variance reduction without V2V communication", status: "PASS" },
];

const navCapabilities = [
  { title: "Photon-Independent", desc: "Navigate in complete darkness using thermal gradients, pressure waves, and molecular composition. Detection range 50-100m with zero photons.", color: "#2AA198" },
  { title: "Molecular Memory", desc: "Roads remember. Exhaust trails persist for hours in the boundary layer, encoding optimal paths discovered by millions of drivers. The membrane reads this directly.", color: "#D4AF37" },
  { title: "Predictive Hazard", desc: "Detect braking 150-290ms before brake lights illuminate (4.5-8.7m at highway speed). Throttle lift changes exhaust composition at sound speed.", color: "#C6A962" },
  { title: "Around-Corner", desc: "Detect hidden vehicles 10-20 seconds before visual contact via exhaust plume diffusion around obstacles. Turbulent diffusion D ≈ 1 m²/s.", color: "#58E6D9" },
  { title: "Sweet Spot", desc: "The cumulative exhaust trail C(x,y) is the probability distribution of optimal paths. P_opt(s) = argmax_y C(x(s), y). The trail IS the solved optimization.", color: "#2AA198" },
  { title: "Emergent Convoy", desc: "Vehicles follow molecular trails, creating self-reinforcing paths. Above critical density ρ_c ≈ 10 veh/km, spontaneous convoy formation with 20-40% fuel savings.", color: "#D4AF37" },
];

export default function Navigation() {
  return (
    <>
      <Head>
        <title>Navigation | Vesicle</title>
        <meta name="description" content="Molecular navigation — superhuman perception through atmospheric computation." />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        {/* Map Section */}
        <section className="w-full h-[70vh] relative">
          <NavigationMap />
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-dark to-transparent h-24 pointer-events-none" />
          <div className="absolute top-4 left-4 bg-dark/80 backdrop-blur-sm rounded-lg p-3 border border-light/10 z-10">
            <div className="text-xs text-gold uppercase tracking-wider font-bold">Munich · Live</div>
            <div className="text-xs text-light/40 mt-1">Molecular trail visualization</div>
          </div>
        </section>

        <Layout>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-5xl mx-auto">
            <h1 className="text-5xl font-bold text-center mb-4 md:text-4xl">Molecular Navigation</h1>
            <p className="text-light/50 text-center mb-12 text-lg max-w-2xl mx-auto">
              The atmosphere remembers. Every vehicle leaves a molecular trail that persists for hours.
              The membrane reads this collective intelligence directly.
            </p>

            {/* Capabilities */}
            <section className="mb-16">
              <div className="grid grid-cols-3 gap-4 md:grid-cols-2 sm:grid-cols-1">
                {navCapabilities.map((cap, i) => (
                  <motion.div key={cap.title} {...fadeIn} transition={{ delay: i * 0.08 }} className="p-5 border border-light/10 rounded-xl">
                    <div className="font-bold mb-2" style={{ color: cap.color }}>{cap.title}</div>
                    <div className="text-sm text-light/50 leading-relaxed">{cap.desc}</div>
                  </motion.div>
                ))}
              </div>
            </section>

            {/* Validation Results */}
            <section className="mb-16">
              <h2 className="text-2xl font-bold text-center mb-8">Validation Results</h2>
              <div className="grid grid-cols-5 gap-3 md:grid-cols-2 sm:grid-cols-1">
                {experiments.map((exp, i) => (
                  <motion.div key={exp.name} {...fadeIn} transition={{ delay: i * 0.08 }} className="p-4 border border-light/10 rounded-xl text-center">
                    <div className="text-2xl font-bold text-gold">{exp.result}</div>
                    <div className="text-sm font-bold text-light/70 mt-1">{exp.name}</div>
                    <div className="text-xs text-light/40 mt-2">{exp.desc}</div>
                    <div className="text-xs mt-2 px-2 py-0.5 rounded-full bg-green-900/30 text-green-400 inline-block">{exp.status}</div>
                  </motion.div>
                ))}
              </div>
            </section>

            {/* The Invisible Road */}
            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-4">The Invisible Road</h2>
              <div className="text-light/60 leading-relaxed space-y-4 max-w-3xl">
                <p>
                  Vehicles do not need lane markings. The molecular trail defines the road. On a snow-covered highway,
                  where cameras see nothing and LiDAR sees only white, the membrane detects exhaust trails persisting
                  in the boundary layer under the snow.
                </p>
                <p>
                  This works on unmarked roads (developing countries), informal roads (dirt tracks), emergency
                  situations (floods, sandstorms), and even off-road terrain where previous vehicles have left trails.
                </p>
                <p>
                  Hazards are encoded as gaps in the molecular trail. Drivers collectively avoid potholes, ice patches,
                  and oil slicks — their avoidance creates absence in the exhaust distribution. The membrane navigates
                  around hazards it has never seen, guided by the collective intelligence of every driver who came before.
                </p>
              </div>
            </section>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

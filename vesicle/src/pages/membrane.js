import Head from "next/head";
import dynamic from "next/dynamic";
import Layout from "@/components/Layout";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";

const MembraneScene = dynamic(
  () => import("@/components/MembraneShader").then((mod) => mod.MembraneScene),
  { ssr: false, loading: () => <div className="w-full h-[60vh] bg-light/5 rounded-2xl animate-pulse flex items-center justify-center text-light/30">Loading membrane visualisation...</div> }
);

const AirflowScene = dynamic(
  () => import("@/components/AirflowScene"),
  { ssr: false, loading: () => <div className="w-full h-[60vh] bg-light/5 rounded-2xl animate-pulse flex items-center justify-center text-light/30">Loading airflow model...</div> }
);

const fadeUp = {
  initial: { y: 40, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

const signalChain = [
  {
    step: 1, title: "Atmospheric O\u2082",
    param: "~10\u2074 molecules, \u03BE_coh \u2248 14 nm",
    desc: "Phase-locked ensembles encode T, P, chemistry, flow",
    color: "#2AA198",
  },
  {
    step: 2, title: "Lipid Coupling",
    param: "Vibrational FRET + rotational-magnetic",
    desc: "Membrane lipids couple to O\u2082 oscillatory modes",
    color: "#C6A962",
  },
  {
    step: 3, title: "P-N Junction",
    param: "V_bi = 0.78 V, \u03C3 = 5.6\u00D710\u207B\u00B3 S/cm",
    desc: "Biological semiconductor converts oscillatory states to electrical",
    color: "#58E6D9",
  },
  {
    step: 4, title: "BMD Transistors",
    param: "On/off ratio 42.1",
    desc: "Pattern recognition switching, not voltage thresholds",
    color: "#D4AF37",
  },
  {
    step: 5, title: "Tri-Logic Gates",
    param: "AND/OR/XOR simultaneously",
    desc: "Three operations from single S-coordinate input",
    color: "#2AA198",
  },
  {
    step: 6, title: "Categorical ALU",
    param: "Partition-space navigation",
    desc: "Computes categorical address resolution",
    color: "#C6A962",
  },
  {
    step: 7, title: "S-entropy Output",
    param: "(S_k, S_t, S_e) \u2208 [0,1]\u00B3",
    desc: "Complete environmental state in three coordinates",
    color: "#58E6D9",
  },
];

function MetricCard({ value, label, unit = "" }) {
  return (
    <motion.div
      {...fadeUp}
      className="flex flex-col items-center p-6 border border-light/10 rounded-xl"
    >
      <span className="text-3xl font-bold text-primaryDark md:text-2xl">
        {value}
        <span className="text-base font-normal ml-1">{unit}</span>
      </span>
      <span className="text-sm text-light/60 mt-2 text-center">{label}</span>
    </motion.div>
  );
}

function InfoCard({ title, description }) {
  return (
    <motion.div
      {...fadeUp}
      className="p-6 border border-light/10 rounded-2xl bg-dark"
    >
      <h3 className="text-lg font-bold mb-3 text-primaryDark">{title}</h3>
      <p className="text-light/70 text-sm leading-relaxed">{description}</p>
    </motion.div>
  );
}

export default function Membrane() {
  return (
    <>
      <Head>
        <title>Membrane | Vesicle</title>
        <meta
          name="description"
          content="Biological membrane computing surface for autonomous vehicles \u2014 lipid signal transduction, phase-locked O\u2082 ensembles, and GLSL shader visualisation."
        />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="The Car That Feels Its Environment"
            className="mb-8 lg:!text-5xl sm:!text-3xl"
          />

          <p className="text-center text-light/70 max-w-3xl mx-auto mb-12 text-lg md:text-base">
            A biological lipid membrane coats the vehicle surface. Each lipid oscillates at 10{"\u00B9\u00B9"} Hz —
            functioning as a processor by the oscillator-processor duality. The membrane simultaneously
            senses, computes, and processes environmental state through phase-locked molecular ensembles.
          </p>

          {/* 3D Membrane Visualisation */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="w-full h-[60vh] md:h-[40vh] rounded-2xl overflow-hidden border border-light/10 mb-16"
          >
            <MembraneScene />
          </motion.div>

          {/* Signal Transduction Chain */}
          <motion.div {...fadeUp} className="mb-16">
            <h2 className="text-4xl font-bold text-center mb-4 lg:text-3xl">
              Signal Transduction Chain
            </h2>
            <p className="text-light/50 text-center mb-10 max-w-2xl mx-auto">
              Seven stages transform atmospheric molecular state into categorical S-entropy coordinates.
            </p>

            {/* Desktop: grid layout with arrows */}
            <div className="hidden md:hidden lg:grid grid-cols-7 gap-2 items-start">
              {signalChain.map((step, i) => (
                <div key={step.step} className="flex items-start gap-2">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.08 }}
                    className="flex-1 border border-light/10 rounded-xl p-4 text-center hover:border-light/25 transition-all"
                  >
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold mx-auto mb-3"
                      style={{ backgroundColor: step.color + "20", color: step.color }}
                    >
                      {step.step}
                    </div>
                    <div className="font-bold text-sm mb-1">{step.title}</div>
                    <div className="text-xs font-semibold mb-2" style={{ color: step.color }}>
                      {step.param}
                    </div>
                    <div className="text-xs text-light/50 leading-relaxed">{step.desc}</div>
                  </motion.div>
                </div>
              ))}
            </div>

            {/* Tablet / Mobile: vertical stack with arrows */}
            <div className="lg:hidden space-y-2">
              {signalChain.map((step, i) => (
                <div key={step.step}>
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.06 }}
                    className="flex items-center gap-4 border border-light/10 rounded-xl p-4 hover:border-light/25 transition-all"
                  >
                    <div
                      className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0"
                      style={{ backgroundColor: step.color + "20", color: step.color }}
                    >
                      {step.step}
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-sm">{step.title}</div>
                      <div className="text-xs font-semibold" style={{ color: step.color }}>
                        {step.param}
                      </div>
                      <div className="text-xs text-light/50 mt-1">{step.desc}</div>
                    </div>
                  </motion.div>
                  {i < signalChain.length - 1 && (
                    <div className="flex justify-center py-1">
                      <span className="text-light/20 text-lg">{"\u2193"}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Key Metrics */}
          <div className="grid grid-cols-4 gap-6 mb-16 md:grid-cols-2 sm:grid-cols-1">
            <MetricCard value="10\u00B2\u2078" unit="ops/s" label="Membrane processing rate" />
            <MetricCard value="10\u00B3\u00B3\u00D7" label="Bandwidth vs ensemble methods" />
            <MetricCard value="18/18" label="Validation tests passing" />
            <MetricCard value="0" label="Free parameters (all from physics)" />
          </div>

          {/* How It Works */}
          <AnimatedText
            text="How the Membrane Works"
            className="mb-8 !text-4xl lg:!text-3xl"
          />

          <div className="grid grid-cols-3 gap-6 mb-16 lg:grid-cols-1">
            <InfoCard
              title="Phase-Locked O\u2082 Ensembles"
              description="~10\u2074 atmospheric O\u2082 molecules form coherent clusters via Van der Waals and paramagnetic coupling (\u03BE \u2248 14 nm). Their phase structure encodes temperature, pressure, chemistry, flow, viscosity, and electromagnetic fields \u2014 complete environmental state in molecular phase."
            />
            <InfoCard
              title="Membrane-O\u2082 Coupling"
              description="The lipid membrane couples to O\u2082 ensembles via vibrational FRET and rotational-magnetic interactions. Information transfers transitively: membrane \u2194 O\u2082 \u2194 road surface, buildings, other vehicles, weather. The membrane senses everything the atmosphere touches."
            />
            <InfoCard
              title="Signal Transduction Chain"
              description="Environmental input propagates through the 7-component biological integrated circuit: BMD transistors (pattern recognition) \u2192 tri-dimensional logic gates (AND/OR/XOR simultaneously) \u2192 virtual ALU \u2192 S-dictionary memory \u2192 S-entropy coordinate output."
            />
          </div>

          {/* Biological Semiconductor */}
          <AnimatedText
            text="Biological Semiconductor"
            className="mb-8 !text-4xl lg:!text-3xl"
          />

          <div className="grid grid-cols-2 gap-6 mb-16 md:grid-cols-1">
            <InfoCard
              title="P-N Junction"
              description="Oscillatory holes (P-type, p = 2.80 \u00D7 10\u00B9\u00B2 cm\u207B\u00B3) and molecular carriers (N-type, n = 1.12 \u00D7 10\u00B9\u00B2 cm\u207B\u00B3) form a biological junction with built-in potential V_bi = 0.78 V and rectification ratio > 32,000. Conductivity: \u03C3 = 5.6 \u00D7 10\u207B\u00B3 S/cm."
            />
            <InfoCard
              title="BMD Transistors"
              description="Biological Maxwell Demon transistors switch on pattern recognition, not voltage thresholds. The gate recognises phase-locked oscillatory signatures in S-entropy space. Clock frequency: 758 Hz (ATP-driven). Coherence time: 10 ms. Fidelity: > 85%."
            />
            <InfoCard
              title="Tri-Dimensional Logic"
              description="AND, OR, and XOR computed simultaneously from the same S-entropy input \u2014 each gate operates on a different entropy dimension (S_k, S_t, S_e). 100% truth table accuracy with 58% component reduction versus NAND-based implementations."
            />
            <InfoCard
              title="Opacity Independence"
              description="Categorical distance is independent of optical opacity: \u2202d_cat/\u2202\u03C4 = 0. The membrane sees through fog, rain, snow, and darkness because molecular phase-locking doesn't depend on photon propagation. Bad weather provides more information, not less."
            />
          </div>

          {/* Airflow Boundary Layer */}
          <motion.div {...fadeUp} className="w-full mb-16">
            <h2 className="text-3xl font-bold mb-4 text-center">
              Atmospheric Boundary Layer
            </h2>
            <p className="text-light/60 text-center mb-6 max-w-2xl mx-auto">
              The membrane reads the airflow boundary layer around the vehicle. Streamlines encode
              velocity, pressure, and molecular composition — the car&apos;s aerodynamic wake is its
              communication signature to other membrane vehicles.
            </p>
            <AirflowScene
              modelPath="/model/airshaper_demo_beta__3d_streamlines.glb"
              height="50vh"
            />
          </motion.div>

          {/* The Counterintuitive Result */}
          <motion.div
            {...fadeUp}
            className="w-full p-8 rounded-2xl border-2 border-primaryDark bg-primaryDark/5 text-center mb-16"
          >
            <h3 className="text-2xl font-bold mb-4 text-primaryDark">
              The Counterintuitive Result
            </h3>
            <p className="text-lg text-light/80 max-w-2xl mx-auto">
              Adverse weather <span className="font-bold text-primaryDark">enhances</span> membrane
              sensing. Rain increases molecular density. Fog enriches phase-locked ensembles.
              Snow modifies surface thermal coupling. Every condition that blinds conventional sensors
              provides <span className="italic">more</span> information to the membrane.
            </p>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

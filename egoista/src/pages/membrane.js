import Head from "next/head";
import dynamic from "next/dynamic";
import Layout from "@/components/Layout";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";

const MembraneScene = dynamic(
  () => import("@/components/MembraneShader").then((mod) => mod.MembraneScene),
  { ssr: false, loading: () => <div className="w-full h-[60vh] bg-dark/10 dark:bg-light/5 rounded-2xl animate-pulse flex items-center justify-center text-dark/30 dark:text-light/30">Loading membrane visualisation...</div> }
);

const fadeUp = {
  initial: { y: 40, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

function MetricCard({ value, label, unit = "" }) {
  return (
    <motion.div
      {...fadeUp}
      className="flex flex-col items-center p-6 border border-dark/10 dark:border-light/10 rounded-xl"
    >
      <span className="text-3xl font-bold text-primary dark:text-primaryDark md:text-2xl">
        {value}
        <span className="text-base font-normal ml-1">{unit}</span>
      </span>
      <span className="text-sm text-dark/60 dark:text-light/60 mt-2 text-center">{label}</span>
    </motion.div>
  );
}

function InfoCard({ title, description }) {
  return (
    <motion.div
      {...fadeUp}
      className="p-6 border-2 border-solid border-dark dark:border-light rounded-2xl bg-light dark:bg-dark"
    >
      <h3 className="text-lg font-bold mb-3 text-primary dark:text-primaryDark">{title}</h3>
      <p className="text-dark/80 dark:text-light/80 text-sm leading-relaxed">{description}</p>
    </motion.div>
  );
}

export default function Membrane() {
  return (
    <>
      <Head>
        <title>Membrane | Egoista</title>
        <meta
          name="description"
          content="Biological membrane computing surface for autonomous vehicles — lipid signal transduction, phase-locked O₂ ensembles, and GLSL shader visualisation."
        />
      </Head>
      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="The Car That Feels Its Environment"
            className="mb-8 lg:!text-5xl sm:!text-3xl"
          />

          <p className="text-center text-dark/70 dark:text-light/70 max-w-3xl mx-auto mb-12 text-lg md:text-base">
            A biological lipid membrane coats the vehicle surface. Each lipid oscillates at 10¹¹ Hz —
            functioning as a processor by the oscillator-processor duality. The membrane simultaneously
            senses, computes, and processes environmental state through phase-locked molecular ensembles.
          </p>

          {/* 3D Membrane Visualisation */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="w-full h-[60vh] md:h-[40vh] rounded-2xl overflow-hidden border border-dark/10 dark:border-light/10 mb-16"
          >
            <MembraneScene />
          </motion.div>

          {/* Key Metrics */}
          <div className="grid grid-cols-4 gap-6 mb-16 md:grid-cols-2 sm:grid-cols-1">
            <MetricCard value="10²⁸" unit="ops/s" label="Membrane processing rate" />
            <MetricCard value="10³³×" label="Bandwidth vs ensemble methods" />
            <MetricCard value="13/13" label="Validation tests passing" />
            <MetricCard value="0" label="Free parameters (all from physics)" />
          </div>

          {/* How It Works */}
          <AnimatedText
            text="How the Membrane Works"
            className="mb-8 !text-4xl lg:!text-3xl"
          />

          <div className="grid grid-cols-3 gap-6 mb-16 lg:grid-cols-1">
            <InfoCard
              title="Phase-Locked O₂ Ensembles"
              description="~10⁴ atmospheric O₂ molecules form coherent clusters via Van der Waals and paramagnetic coupling (ξ ≈ 14 nm). Their phase structure encodes temperature, pressure, chemistry, flow, viscosity, and electromagnetic fields — complete environmental state in molecular phase."
            />
            <InfoCard
              title="Membrane-O₂ Coupling"
              description="The lipid membrane couples to O₂ ensembles via vibrational FRET and rotational-magnetic interactions. Information transfers transitively: membrane ↔ O₂ ↔ road surface, buildings, other vehicles, weather. The membrane senses everything the atmosphere touches."
            />
            <InfoCard
              title="Signal Transduction Chain"
              description="Environmental input propagates through the 7-component biological integrated circuit: BMD transistors (pattern recognition) → tri-dimensional logic gates (AND/OR/XOR simultaneously) → virtual ALU → S-dictionary memory → S-entropy coordinate output."
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
              description="Oscillatory holes (P-type, p = 2.80 × 10¹² cm⁻³) and molecular carriers (N-type, n = 1.12 × 10¹² cm⁻³) form a biological junction with built-in potential V_bi = 0.78 V and rectification ratio > 32,000. Conductivity: σ = 5.6 × 10⁻³ S/cm."
            />
            <InfoCard
              title="BMD Transistors"
              description="Biological Maxwell Demon transistors switch on pattern recognition, not voltage thresholds. The gate recognises phase-locked oscillatory signatures in S-entropy space. Clock frequency: 758 Hz (ATP-driven). Coherence time: 10 ms. Fidelity: > 85%."
            />
            <InfoCard
              title="Tri-Dimensional Logic"
              description="AND, OR, and XOR computed simultaneously from the same S-entropy input — each gate operates on a different entropy dimension (S_k, S_t, S_e). 100% truth table accuracy with 58% component reduction versus NAND-based implementations."
            />
            <InfoCard
              title="Opacity Independence"
              description="Categorical distance is independent of optical opacity: ∂d_cat/∂τ = 0. The membrane sees through fog, rain, snow, and darkness because molecular phase-locking doesn't depend on photon propagation. Bad weather provides more information, not less."
            />
          </div>

          {/* The Counterintuitive Result */}
          <motion.div
            {...fadeUp}
            className="w-full p-8 rounded-2xl border-2 border-primary dark:border-primaryDark bg-primary/5 dark:bg-primaryDark/5 text-center mb-16"
          >
            <h3 className="text-2xl font-bold mb-4 text-primary dark:text-primaryDark">
              The Counterintuitive Result
            </h3>
            <p className="text-lg text-dark/80 dark:text-light/80 max-w-2xl mx-auto">
              Adverse weather <span className="font-bold text-primary dark:text-primaryDark">enhances</span> membrane
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

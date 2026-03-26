import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import AnimatedText from "@/components/AnimatedText";
import { motion } from "framer-motion";

const fadeUp = {
  initial: { y: 40, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.6 },
};

/* ------------------------------------------------------------------ */
/*  Gusheshe Data                                                      */
/* ------------------------------------------------------------------ */

const gushesheComponents = [
  {
    name: "Point",
    color: "#2AA198",
    desc: "Semantic units carrying confidence (0\u20131), validity windows, and categories: Observation, Inference, Prediction, Safety. The atomic unit of reasoning.",
  },
  {
    name: "Resolution",
    color: "#D4AF37",
    desc: "A debate platform that combines affirmations and contentions to reach justified conclusions. Each resolution produces a confidence-weighted output.",
  },
  {
    name: "Certificate",
    color: "#58E6D9",
    desc: "Pre-compiled, verifiable execution units. A certificate guarantees that a resolution was computed correctly and can be replayed for audit.",
  },
  {
    name: "Strategies",
    color: "#C6A962",
    desc: "Five resolution strategies: Bayesian (prior-weighted), MaximumLikelihood (data-driven), Conservative (worst-case), Exploratory (information-seeking), Adaptive (context-switching).",
  },
];

const gushesheMetrics = [
  { value: "100ms", label: "Resolution timeout" },
  { value: "0.65", label: "Confidence threshold" },
  { value: "10", label: "Concurrent resolutions" },
  { value: "3", label: "Reasoning paradigms" },
];

/* ------------------------------------------------------------------ */
/*  Ruzende Data                                                       */
/* ------------------------------------------------------------------ */

const ruzendeConnections = [
  {
    from: "Gusheshe",
    to: "Izinyoka",
    role: "Metacognition",
    standard: "50ms",
    emergency: "10ms",
    color: "#2AA198",
  },
  {
    from: "Sighthound",
    to: "Fusion",
    role: "Sensor data \u2192 categorical state",
    standard: "20ms",
    emergency: "5ms",
    color: "#D4AF37",
  },
  {
    from: "Combine Harvester",
    to: "Integration",
    role: "Data transformation",
    standard: "30ms",
    emergency: "15ms",
    color: "#58E6D9",
  },
];

const messagePatterns = [
  {
    name: "Standard",
    color: "#2AA198",
    desc: "Normal priority. Queued and delivered within the module's standard timing window. Retries on timeout.",
  },
  {
    name: "Uncertainty",
    color: "#D4AF37",
    desc: "Confidence below threshold. Triggers additional resolution rounds and requests corroborating data from adjacent modules.",
  },
  {
    name: "Emergency",
    color: "#E74C3C",
    desc: "Safety-critical. Bypasses queue, delivered within emergency timing window. All non-emergency traffic yields.",
  },
];

const ruzendeSample = `protocol SensorFusion {
  module Sighthound -> Gusheshe {
    timing: standard 20ms, emergency 5ms
    payload: CategoricalState
    retry: 3 attempts, exponential backoff
    fallback: last_known_state
  }

  module Gusheshe -> Izinyoka {
    timing: standard 50ms, emergency 10ms
    payload: Resolution<Certificate>
    on_uncertainty: request_corroboration
    on_timeout: degrade_gracefully
  }

  flow Emergency {
    priority: maximum
    preempt: all_standard
    validate: safety_certificate_required
  }
}`;

/* ------------------------------------------------------------------ */
/*  Pipeline Data                                                      */
/* ------------------------------------------------------------------ */

const pipelineSteps = [
  { name: "Membrane", timing: "continuous", color: "#2AA198", desc: "Lipid bilayer senses atmospheric molecular state" },
  { name: "Sighthound", timing: "~20ms", color: "#D4AF37", desc: "Sensor fusion into categorical S-entropy coordinates" },
  { name: "Gusheshe", timing: "~100ms", color: "#58E6D9", desc: "Hybrid resolution: logical + fuzzy + Bayesian inference" },
  { name: "Ruzende", timing: "~50ms", color: "#C6A962", desc: "Inter-module coordination and protocol enforcement" },
  { name: "Vehicle Control", timing: "<200ms total", color: "#2AA198", desc: "Actuator commands with safety certificates" },
];

/* ------------------------------------------------------------------ */
/*  Reusable Components                                                */
/* ------------------------------------------------------------------ */

function Card({ children, className = "" }) {
  return (
    <motion.div
      {...fadeUp}
      className={`p-6 border border-light/10 rounded-2xl bg-dark ${className}`}
    >
      {children}
    </motion.div>
  );
}

function SectionLabel({ color, children }) {
  return (
    <span
      className="text-xs font-mono px-3 py-1 rounded-full inline-block mb-4 font-semibold tracking-wider uppercase"
      style={{ backgroundColor: color + "18", color }}
    >
      {children}
    </span>
  );
}

function MetricPill({ value, label }) {
  return (
    <div className="flex flex-col items-center p-4 border border-light/10 rounded-xl">
      <span className="text-2xl font-bold" style={{ color: "#D4AF37" }}>{value}</span>
      <span className="text-xs text-light/50 mt-1 text-center">{label}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function Systems() {
  return (
    <>
      <Head>
        <title>Systems | Vesicle</title>
        <meta
          name="description"
          content="Gusheshe hybrid resolution engine and Ruzende inter-module communication DSL \u2014 the software connecting membrane sensing to driving decisions."
        />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        <Layout className="pt-16">
          {/* Header */}
          <AnimatedText text="Core Systems" className="mb-4 lg:!text-5xl sm:!text-3xl" />
          <p className="text-center text-light/50 max-w-3xl mx-auto mb-20 text-lg md:text-base">
            The software that connects membrane sensing to driving decisions
          </p>

          {/* ============================================================ */}
          {/*  SECTION 1: Gusheshe                                         */}
          {/* ============================================================ */}
          <motion.div {...fadeUp} className="mb-24">
            <SectionLabel color="#2AA198">Engine</SectionLabel>
            <h2 className="text-4xl font-bold mb-2 lg:text-3xl">Gusheshe</h2>
            <p className="text-light/40 text-lg mb-2 italic">Hybrid Resolution Engine</p>
            <p className="text-light/60 mb-8 max-w-2xl" style={{ color: "#2AA198" }}>
              Logical + Fuzzy + Bayesian inference in 10&ndash;100ms
            </p>
            <p className="text-light/70 mb-10 max-w-3xl leading-relaxed">
              Gusheshe is a real-time decision engine that combines three reasoning paradigms into a single
              resolution framework. Every driving decision is modelled as a debate: affirmations and contentions
              compete under configurable strategies, producing confidence-weighted certificates that downstream
              modules can verify independently.
            </p>

            {/* Component cards */}
            <div className="grid grid-cols-2 gap-6 mb-10 md:grid-cols-1">
              {gushesheComponents.map((c) => (
                <Card key={c.name}>
                  <h3 className="text-lg font-bold mb-2" style={{ color: c.color }}>{c.name}</h3>
                  <p className="text-light/60 text-sm leading-relaxed">{c.desc}</p>
                </Card>
              ))}
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-4 gap-4 mb-10 md:grid-cols-2 sm:grid-cols-1">
              {gushesheMetrics.map((m) => (
                <MetricPill key={m.label} value={m.value} label={m.label} />
              ))}
            </div>

            {/* Architecture flow */}
            <motion.div
              {...fadeUp}
              className="border border-light/10 rounded-2xl p-8 bg-light/[0.02]"
            >
              <div className="text-xs text-light/40 uppercase tracking-wider mb-4">Architecture Flow</div>
              <div className="flex items-center justify-center gap-3 flex-wrap text-sm">
                {["Sensor Input", "Point Construction", "Resolution Debate", "Strategy Selection", "Certificate Output"].map((step, i) => (
                  <span key={step} className="flex items-center gap-3">
                    <span className="px-4 py-2 border border-light/15 rounded-lg bg-dark text-light/80 font-medium">
                      {step}
                    </span>
                    {i < 4 && <span className="text-light/30">{"\u2192"}</span>}
                  </span>
                ))}
              </div>
            </motion.div>
          </motion.div>

          {/* ============================================================ */}
          {/*  SECTION 2: Ruzende                                          */}
          {/* ============================================================ */}
          <motion.div {...fadeUp} className="mb-24">
            <SectionLabel color="#D4AF37">Communication</SectionLabel>
            <h2 className="text-4xl font-bold mb-2 lg:text-3xl">Ruzende</h2>
            <p className="text-light/40 text-lg mb-2 italic">Inter-Module Communication DSL</p>
            <p className="text-light/60 mb-8 max-w-2xl" style={{ color: "#D4AF37" }}>
              Declarative protocols between system modules
            </p>
            <p className="text-light/70 mb-10 max-w-3xl leading-relaxed">
              Ruzende is a domain-specific language for defining communication patterns, timing constraints, and
              error recovery between system modules. Every message carries timing guarantees: standard latency for
              normal operation, emergency latency for safety-critical paths.
            </p>

            {/* Module connections */}
            <div className="space-y-4 mb-10">
              {ruzendeConnections.map((c) => (
                <motion.div
                  key={c.from + c.to}
                  {...fadeUp}
                  className="flex items-center gap-4 p-5 border border-light/10 rounded-xl md:flex-col md:items-start"
                >
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <span className="font-bold text-sm" style={{ color: c.color }}>{c.from}</span>
                    <span className="text-light/30">{"\u2194"}</span>
                    <span className="font-bold text-sm" style={{ color: c.color }}>{c.to}</span>
                  </div>
                  <div className="text-light/50 text-sm flex-1">{c.role}</div>
                  <div className="flex gap-4 text-xs flex-shrink-0">
                    <span className="px-3 py-1 rounded-full border border-light/10">
                      Standard: <span className="font-bold text-light/80">{c.standard}</span>
                    </span>
                    <span className="px-3 py-1 rounded-full border border-red-500/30" style={{ color: "#E74C3C" }}>
                      Emergency: <span className="font-bold">{c.emergency}</span>
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Message flow patterns */}
            <div className="grid grid-cols-3 gap-6 mb-10 md:grid-cols-1">
              {messagePatterns.map((p) => (
                <Card key={p.name}>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: p.color }} />
                    <h3 className="font-bold" style={{ color: p.color }}>{p.name}</h3>
                  </div>
                  <p className="text-light/60 text-sm leading-relaxed">{p.desc}</p>
                </Card>
              ))}
            </div>

            {/* Code example */}
            <motion.div {...fadeUp}>
              <div className="text-xs text-light/40 uppercase tracking-wider mb-3">Ruzende Protocol Example</div>
              <pre className="bg-light/[0.03] border border-light/10 rounded-xl p-6 overflow-x-auto text-sm leading-relaxed">
                <code className="text-light/70">{ruzendeSample}</code>
              </pre>
            </motion.div>
          </motion.div>

          {/* ============================================================ */}
          {/*  SECTION 3: How They Connect                                 */}
          {/* ============================================================ */}
          <motion.div {...fadeUp} className="mb-16">
            <SectionLabel color="#58E6D9">Pipeline</SectionLabel>
            <h2 className="text-4xl font-bold mb-2 lg:text-3xl">How They Connect</h2>
            <p className="text-light/60 mb-10 max-w-2xl">
              The full pipeline from atmospheric sensing to vehicle control. Total latency under 200ms.
            </p>

            {/* Pipeline steps */}
            <div className="space-y-3 mb-12">
              {pipelineSteps.map((step, i) => (
                <div key={step.name}>
                  <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.08, duration: 0.5 }}
                    className="flex items-center gap-5 p-5 border border-light/10 rounded-xl hover:border-light/20 transition-all"
                  >
                    <div
                      className="w-11 h-11 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0"
                      style={{ backgroundColor: step.color + "20", color: step.color }}
                    >
                      {i + 1}
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-base">{step.name}</div>
                      <div className="text-sm text-light/50 mt-0.5">{step.desc}</div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <div className="text-lg font-bold" style={{ color: step.color }}>{step.timing}</div>
                    </div>
                  </motion.div>
                  {i < pipelineSteps.length - 1 && (
                    <div className="flex justify-center py-1">
                      <span className="text-light/20 text-lg">{"\u2193"}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Total latency callout */}
            <motion.div
              {...fadeUp}
              className="w-full p-8 rounded-2xl border-2 text-center"
              style={{ borderColor: "#58E6D920", backgroundColor: "#58E6D908" }}
            >
              <div className="text-xs text-light/40 uppercase tracking-wider mb-2">End-to-End Latency</div>
              <div className="text-4xl font-bold mb-2" style={{ color: "#58E6D9" }}>&lt;200ms</div>
              <p className="text-light/60 max-w-lg mx-auto">
                From atmospheric molecular state to actuator command. Sensor fusion ~20ms, resolution ~100ms,
                coordination ~50ms. Every step carries a verifiable certificate.
              </p>
            </motion.div>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

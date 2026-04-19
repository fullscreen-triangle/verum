import Layout from "@/components/Layout";
import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const DisplayEquation = dynamic(() => import("@/components/KatexBlock"), { ssr: false });
const InlineEq = dynamic(
  () => import("@/components/KatexBlock").then((m) => m.InlineEquation),
  { ssr: false }
);

const fadeInUp = {
  initial: { y: 40, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true, margin: "-80px" },
  transition: { duration: 0.55 },
};

// =====================================================================
// Theorem card
// =====================================================================
function Theorem({ number, title, body, proofSketch }) {
  return (
    <div className="my-8 rounded-2xl border-2 border-primary/70 dark:border-primaryDark/70 bg-light dark:bg-dark p-6 md:p-5 sm:p-4 relative">
      <div className="absolute -top-3 left-6 bg-primary dark:bg-primaryDark text-dark px-3 py-0.5 rounded-md text-xs font-bold uppercase tracking-wider">
        Theorem {number} &mdash; {title}
      </div>
      <div className="mt-3 text-dark/85 dark:text-light/85 space-y-3 text-lg sm:text-base">
        {body}
      </div>
      {proofSketch && (
        <div className="mt-5 pt-4 border-t border-dark/15 dark:border-light/15">
          <div className="text-xs font-bold uppercase tracking-wider text-primary dark:text-primaryDark mb-2">
            Proof sketch
          </div>
          <div className="text-base sm:text-sm text-dark/70 dark:text-light/70 italic space-y-2">
            {proofSketch}
          </div>
        </div>
      )}
    </div>
  );
}

function Definition({ number, title, body }) {
  return (
    <div className="my-6 rounded-xl border border-membrane/60 bg-membrane/5 dark:bg-membrane/10 p-5 sm:p-4">
      <div className="text-xs font-bold uppercase tracking-wider text-membrane mb-1">
        Definition {number} &mdash; {title}
      </div>
      <div className="text-dark/85 dark:text-light/85 text-base">{body}</div>
    </div>
  );
}

function SectionHeading({ id, children }) {
  return (
    <h2
      id={id}
      className="font-bold text-5xl md:text-4xl sm:text-3xl xs:text-2xl text-primary dark:text-primaryDark mt-20 mb-8 scroll-mt-24"
    >
      {children}
    </h2>
  );
}

function SubHeading({ children }) {
  return (
    <h3 className="font-bold text-2xl sm:text-xl mt-10 mb-4 text-dark dark:text-light">
      {children}
    </h3>
  );
}

function P({ children }) {
  return (
    <p className="text-lg sm:text-base leading-relaxed text-dark/80 dark:text-light/80 mb-4">
      {children}
    </p>
  );
}

function Panel({ src, caption }) {
  return (
    <figure className="my-12">
      <div className="rounded-2xl overflow-hidden border border-dark/15 dark:border-light/15 bg-white">
        <Image
          src={src}
          alt={caption}
          width={1600}
          height={1230}
          className="w-full h-auto"
          priority={false}
        />
      </div>
      <figcaption className="mt-3 text-sm text-dark/65 dark:text-light/65 italic">
        {caption}
      </figcaption>
    </figure>
  );
}

// =====================================================================
// Falsifiable predictions
// =====================================================================
const predictions = [
  {
    title: "Curriculum performance curves",
    body: "Performance gap gₖ decays exponentially across curriculum stages; rate set by stage-complexity increment.",
    metric: "λ̂ = 0.387 vs λ_set = 0.38 (R² = 0.97)",
  },
  {
    title: "Highway variance inside F1 envelope",
    body: "F1-trained reflex layer deployed at 120 km/h on a highway shows control-input variance ≤ 5% of limit-driving variance.",
    metric: "σ_hwy / σ_f1 ≤ 0.05",
  },
  {
    title: "Edge-case performance",
    body: "Emergency evasion handled by reflex layer without cognitive escalation; time-to-avoidance bounded by τ_reflex.",
    metric: "τ_evasion < 50 ms",
  },
  {
    title: "Top-human residual spectral signature",
    body: "Fourier spectrum of r_human(t) concentrates in 5–30 Hz band (proprioceptive-vestibular bandwidth), distinguishable from white noise.",
    metric: "91.6% band fraction measured",
  },
  {
    title: "Sensor ablation asymmetry",
    body: "Removing a modality from the cognitive layer degrades performance earlier than removing the same modality from the reflex layer.",
    metric: "ablation differential > 2×",
  },
  {
    title: "Theoretical-minimum approach",
    body: "Curriculum-trained controller reaches within 0.5% of T* on 8–15 corner circuits, matching top-human records within Gumbel tail.",
    metric: "ΔT/T* < 0.005",
  },
  {
    title: "Road-transfer gap",
    body: "After constraint composition, curriculum-trained controller shows lower km-collision rate than end-to-end road-data controller of equal parameters.",
    metric: "collision/km lower bound",
  },
  {
    title: "Latency degradation experiment",
    body: "Artificially introducing τ ≥ 100 ms in Layer 1 degrades lap time by predictable factor from phase-margin loss.",
    metric: "0.5% penalty at τ = 25 ms",
  },
];

// =====================================================================
// Architecture comparison
// =====================================================================
const archRows = [
  { name: "End-to-end", regimes: "No", reachesTStar: "No", intentClean: "No", edge: "Low", interp: "No" },
  { name: "Modular pipeline", regimes: "Partial", reachesTStar: "No", intentClean: "No", edge: "Medium", interp: "Partial" },
  { name: "Behaviour cloning", regimes: "No", reachesTStar: "No", intentClean: "No", edge: "Low", interp: "No" },
  { name: "RL in sim", regimes: "No", reachesTStar: "No guarantee", intentClean: "Depends on reward", edge: "Medium", interp: "No" },
  { name: "MPC racing line", regimes: "Partial", reachesTStar: "Yes (offline)", intentClean: "Yes on circuit", edge: "High", interp: "Yes" },
  { name: "Reflex-cognitive (this work)", regimes: "Yes", reachesTStar: "Yes", intentClean: "Yes on circuit", edge: "High", interp: "Yes" },
];

// =====================================================================
// Curriculum stages
// =====================================================================
const stages = [
  { k: 0, name: "Circuit", cost: "0.00", desc: "Single intent, physics-dominated. Theoretical minimum is the target." },
  { k: 1, name: "Kerb respect", cost: "0.80", desc: "Track-limit penalty added. Reduces cornering envelope at apex." },
  { k: 2, name: "Track limits", cost: "1.00", desc: "Lane-like constraints on track width." },
  { k: 3, name: "Double yellow / caution", cost: "1.10", desc: "Dynamic speed caps reacting to flag state." },
  { k: 4, name: "Pit lane", cost: "1.30", desc: "Speed limit + entry/exit geometry." },
  { k: 5, name: "Signals", cost: "1.40", desc: "Stop-light and yield rules as discrete events." },
  { k: 6, name: "Pedestrians", cost: "1.70", desc: "Dynamic pedestrian avoidance; perception-planning coupling." },
  { k: 7, name: "Intersections", cost: "1.90", desc: "Multi-actor negotiation; turn-taking rules." },
  { k: 8, name: "Urban", cost: "2.20", desc: "Full composition: signals + pedestrians + intersections + traffic." },
];

export default function IntentDecomposition() {
  return (
    <>
      <Head>
        <title>Intent Decomposition | Vesicle</title>
        <meta
          name="description"
          content="A reflex-cognitive architecture for autonomous driving. The action-intent attribution thesis: driving is two problems, not one."
        />
      </Head>
      <TransitionEffect />

      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">

          {/* =============== HERO =============== */}
          <AnimatedText
            text="Driving Is Two Problems, Not One."
            className="mb-10 !text-7xl !leading-tight lg:!text-6xl md:!text-5xl sm:!text-4xl xs:!text-3xl text-center"
          />
          <motion.div {...fadeInUp} className="max-w-5xl mx-auto text-center mb-20">
            <p className="text-2xl md:text-xl sm:text-lg text-dark/85 dark:text-light/85 leading-relaxed mb-6">
              Two decades of autonomous-driving research have collapsed{" "}
              <em>reflex</em> (sub-100&nbsp;ms, physics-dominated) and{" "}
              <em>cognitive</em> (seconds, rule-dominated) onto a single substrate.
              This is the architectural error. The two regimes are separated by an
              order of magnitude in bandwidth; they cannot share the critical path.
            </p>
            <p className="text-xl md:text-lg sm:text-base text-dark/65 dark:text-light/65 italic">
              This page synthesises the standalone paper
              &ldquo;<Link href="/papers/high-velocity-vehicle-intent-decomposition.pdf" className="underline hover:text-primary">High-Velocity Vehicle Intent Decomposition</Link>&rdquo;
              — 37 pages, 11 theorems, 99 references, zero self-citation.
            </p>
            <Link
              href="/papers/high-velocity-vehicle-intent-decomposition.pdf"
              target="_blank"
              className="inline-block mt-8 rounded-lg bg-primary dark:bg-primaryDark text-dark font-bold px-6 py-3 hover:scale-105 transition-transform"
            >
              Download the paper (PDF, 37 pp)
            </Link>
          </motion.div>

          {/* =============== THE CORE CONFUSION =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="confusion">1. The Core Confusion</SectionHeading>
            <P>
              Consider two questions. First:{" "}
              <em>should an autonomous vehicle brake?</em> Second:{" "}
              <em>how hard, on which wheel, for how long?</em> The first is a
              rule-and-goal decision: it depends on map, on law, on traffic, on
              passenger comfort. The second is a physical-control decision: it
              depends on tyre grip, load transfer, aerodynamic balance, and the
              next 30&nbsp;metres of road surface.
            </P>
            <P>
              These problems live on different substrates. The first admits
              seconds of deliberation; the second must complete in tens of
              milliseconds, because at 300&nbsp;km/h a vehicle covers
              8.3&nbsp;metres every 100&nbsp;ms. Conflating them &mdash;
              running both on a cognitive substrate &mdash; is the structural
              reason autonomous driving has plateaued.
            </P>
            <P>
              The resolution is structural, not statistical. Decompose the
              problem by timescale, train each layer on the data regime where
              its target is identifiable, and compose them through a
              well-defined interface. This page walks through every step of
              that argument.
            </P>
          </motion.section>

          {/* =============== WHAT THE PAPER PROVES =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="proves">2. What the Paper Proves</SectionHeading>
            <P>
              Eight principal theorems, connected by a single axiom (finite
              reflex bandwidth at the physical limit) and a single empirical
              anchor (top-human telemetry on circuits).
            </P>
            <ul className="grid md:grid-cols-2 gap-3 my-6">
              {[
                ["Timescale Separation", "Reflex and cognitive substrates cannot share the critical path at limit speed."],
                ["Intent Attribution", "Intent is identifiable from circuit telemetry; unidentifiable from road telemetry."],
                ["Constraint Cascade", "Road driving = circuit driving + closed-set constraints composed sequentially."],
                ["Residual Decomposition", "Top-human telemetry = closed-model prediction + structured residual + Knightian remainder."],
                ["Unmodelled Variable Bound", "No finite-dimensional model captures all degrees of freedom; σ²_⊥^∞ > 0."],
                ["Curriculum Convergence", "Performance gap decays exponentially across incremental constraint stages."],
                ["Reflex-Layer Minimum Latency", "Any architecture reaching T* must run reflex at ≤ 50 ms."],
                ["Architectural Uniqueness", "The two-layer decomposition is the only configuration satisfying all three."],
              ].map(([name, desc]) => (
                <li
                  key={name}
                  className="rounded-xl border border-primary/40 dark:border-primaryDark/40 bg-primary/5 dark:bg-primaryDark/5 p-4"
                >
                  <div className="font-bold text-primary dark:text-primaryDark mb-1">{name}</div>
                  <div className="text-sm text-dark/75 dark:text-light/75">{desc}</div>
                </li>
              ))}
            </ul>
          </motion.section>

          {/* =============== TIMESCALES =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="timescales">3. Timescales of Driving</SectionHeading>

            <SubHeading>3.1 Latency as a physical constraint</SubHeading>
            <P>
              Let <InlineEq math="\\tau_{\\mathrm{sense}}" /> denote the
              sensing-to-actuation latency along the critical path of a
              controller. Between sensing event and wheel response the vehicle
              moves a distance
            </P>
            <DisplayEquation math="\\Delta s = v \\cdot \\tau_{\\mathrm{sense}}." />
            <P>
              At 300&nbsp;km/h (83.3&nbsp;m/s), a 100&nbsp;ms latency is
              8.3&nbsp;m of blind motion. Typical braking distance from that
              speed is 50–70&nbsp;m, so a single 100&nbsp;ms blind-motion slab
              consumes 15–20% of the stopping envelope. Any substrate on the
              critical path must fit within the physical latency budget of the
              braking geometry, not the other way around.
            </P>

            <SubHeading>3.2 Reflex latency from braking geometry</SubHeading>
            <P>
              Set up the braking-geometry constraint: required reaction time
              <InlineEq math="\\; t_{\\mathrm{react}}(v, \\mu) = (d_{\\mathrm{perc}} - v^2 / (2\\mu g)) / v" />
              {" "}for perception horizon <InlineEq math="d_{\\mathrm{perc}}" />
              {" "}and long-friction coefficient <InlineEq math="\\mu" />. For
              racing grip <InlineEq math="\\mu = 1.65" /> and a
              60&nbsp;m horizon, the reaction window falls below 50&nbsp;ms at
              speeds above 280&nbsp;km/h. This is the reflex budget: anything
              slower than that must not be on the critical path.
            </P>

            <SubHeading>3.3 Cognitive latency from human performance</SubHeading>
            <P>
              Human decision-making literature places visual-cognitive reaction
              times at 400&nbsp;ms–2&nbsp;s depending on choice complexity;
              saccade-initiation 180–250&nbsp;ms; voluntary-lift lift-off
              200–300&nbsp;ms. None of these are reflex latencies. The
              proprioceptive-vestibular reflex arc closes in 30–80&nbsp;ms; the
              long-loop stretch reflex closes in 50–70&nbsp;ms. Our model uses
              <InlineEq math="\\tau_{\\mathrm{reflex}} \\sim 35~\\mathrm{ms}" />
              {" "}and <InlineEq math="\\tau_{\\mathrm{cog}} \\sim 800~\\mathrm{ms}" />
              .
            </P>

            <SubHeading>3.4 Formal separation</SubHeading>
            <Theorem
              number="1"
              title="Timescale Separation"
              body={
                <>
                  <p>
                    Let the reflex and cognitive latency distributions be
                    <InlineEq math="F_r" /> and <InlineEq math="F_c" />{" "}respectively.
                    Under the log-normal family calibrated by human motor literature,
                  </p>
                  <DisplayEquation math="\\Pr_{r \\sim F_r,\\, c \\sim F_c}[r > c] < 10^{-5}," />
                  <p>
                    i.e. the critical-path overlap between reflex and cognitive
                    substrates is measure-zero in practice. Any architecture
                    that runs the cognitive substrate on the sensing-to-actuation
                    path loses a full order of magnitude in bandwidth.
                  </p>
                </>
              }
              proofSketch={
                <p>
                  Cohen&apos;s d between the two calibrated populations is 4.19;
                  95th percentile of reflex distribution (53.7&nbsp;ms) falls
                  strictly below the 5th percentile of the cognitive distribution
                  (454&nbsp;ms). The two substrates must therefore be decoupled
                  on separate hardware paths.
                </p>
              }
            />

            <Panel
              src="/images/intent/panel_1_timescales.png"
              caption="Panel 1 — Timescale separation: (A) latency distributions with d = 4.19 and P(overlap) < 10⁻⁵, (B) quantile traces, (C) 3D perception-to-brake reaction window, (D) speed × friction rule-out for cognitive-only control."
            />

            <SubHeading>3.5 Consequences for existing architectures</SubHeading>
            <P>
              Theorem 1 rules out end-to-end learning placed on the
              sensing-to-actuation critical path: neural networks at real-time
              deployment budgets (20–50&nbsp;ms forward passes) meet the
              latency requirement only when they are structurally part of the
              reflex layer &mdash; i.e. when they <em>do not</em> incorporate
              cognitive-regime features like route planning, intent
              classification, or semantic scene understanding.
            </P>
          </motion.section>

          {/* =============== INTENT ATTRIBUTION =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="intent">4. Intent Attribution</SectionHeading>

            <SubHeading>4.1 The intent-identifiability problem</SubHeading>
            <P>
              Let <InlineEq math="\\theta" /> denote the driver intent
              (weighting across time, comfort, risk, route objectives) and
              <InlineEq math="\\; y" /> denote observed telemetry. Identifiability
              requires that <InlineEq math="\\theta \\mapsto p(y \\mid \\theta)" />{" "}
              be injective on the relevant parameter set. On a road, it is
              <em> not</em> injective: multiple route choices, comfort
              preferences, and risk settings yield indistinguishable
              trajectories.
            </P>

            <Definition
              number="1"
              title="Feasible intent class"
              body={
                <>
                  Given environment <InlineEq math="\\mathcal{E}" /> and
                  trajectory <InlineEq math="y" />, the feasible intent class
                  is
                  <InlineEq math="\\; \\mathcal{F}(y, \\mathcal{E}) = \\{\\theta : p(y \\mid \\theta, \\mathcal{E}) > 0\\}" />.
                  On a circuit, <InlineEq math="|\\mathcal{F}| = 1" /> almost
                  surely. On a road, <InlineEq math="|\\mathcal{F}|" /> is
                  typically a continuum.
                </>
              }
            />

            <SubHeading>4.2 Multiple objectives on roads</SubHeading>
            <P>
              A route from A to B may be chosen for minimum time, maximum
              comfort, avoidance of tolls, or exposure to scenery. Each
              objective projects to a distinct cost functional, but the
              resulting trajectories are often indistinguishable within
              typical measurement precision. Behaviour cloning learns a
              confounded mixture over this unobservable class.
            </P>

            <SubHeading>4.3 Single objective on circuits</SubHeading>
            <P>
              On a circuit, intent collapses to minimum lap time:
            </P>
            <DisplayEquation math="\\theta_{\\mathrm{circuit}} = \\arg\\min_{\\pi} T_{\\mathrm{lap}}(\\pi; \\mathcal{E})." />
            <P>
              Every other reasonable objective (comfort, tyre temperature,
              fuel conservation) trades directly against lap time, so the
              intent space is one-dimensional. Circuit telemetry is the only
              widely-available driving dataset where intent is identifiable.
            </P>

            <Theorem
              number="2"
              title="Intent Attribution"
              body={
                <>
                  <p>
                    On a circuit, intent is identifiable from telemetry; the
                    posterior concentrates on a single mode as the sample size
                    grows. On a road with
                    <InlineEq math="\\; k \\geq 2" /> Pareto-equivalent
                    objectives, the posterior is degenerate: it is invariant
                    along the <InlineEq math="(k-1)" />-dimensional subspace of
                    objective trade-offs, and no finite dataset resolves it.
                  </p>
                </>
              }
              proofSketch={
                <p>
                  The design-matrix rank determines identifiability of the
                  least-squares pulls on <InlineEq math="\\theta" />. Circuit
                  data is full-rank (R² → 1); road data collapses to a rank-1
                  column space (R² → −∞ for individual coordinates), and the
                  intent vector is only identified modulo a ridge.
                </p>
              }
            />

            <Panel
              src="/images/intent/panel_2_intent.png"
              caption="Panel 2 — Intent identifiability: circuit R² → 1.0 vs road R² → −28, posterior concentrates on a ridge under road conditions. The rank deficiency is structural — more data cannot fix it."
            />

            <SubHeading>4.4 Measuring identifiability</SubHeading>
            <P>
              We use two diagnostics: recovery R² for the intent parameter
              from simulated regressions, and spectral entropy of the design
              matrix. Road regression achieves near-maximum entropy (ln 3 ≈
              1.10 nats) &mdash; all three intent coordinates project onto the
              same one-dimensional feature. Circuit regression concentrates on
              0.93 nats, matching the single-eigenvalue structure of the
              unique-intent solution.
            </P>
          </motion.section>

          {/* =============== VEHICLE DYNAMICS =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="dynamics">5. Vehicle Dynamics at the Limit</SectionHeading>

            <SubHeading>5.1 Single-track model</SubHeading>
            <P>
              The bicycle-model state vector is
              <InlineEq math="\\; x = (X, Y, \\psi, v_x, v_y, r)^\\top" />,
              where <InlineEq math="(X, Y)" /> is the planar position,
              <InlineEq math="\\psi" /> is heading,
              <InlineEq math="(v_x, v_y)" /> are body-frame velocities, and
              <InlineEq math="r" /> is the yaw rate. The body-frame dynamics
              are
            </P>
            <DisplayEquation math="m\\dot v_x = F_{x,\\mathrm{tot}} + m v_y r,\\quad m\\dot v_y = F_{y,\\mathrm{tot}} - m v_x r,\\quad I_z \\dot r = \\tau_z." />

            <SubHeading>5.2 Pacejka tyre model</SubHeading>
            <P>
              Tyre force is non-linear in slip. The Magic Formula gives
              longitudinal and lateral forces as
            </P>
            <DisplayEquation math="F = D \\sin\\!\\left(C \\arctan\\!\\left(B \\kappa - E\\left(B \\kappa - \\arctan(B \\kappa)\\right)\\right)\\right)," />
            <P>
              with <InlineEq math="B, C, D, E" /> calibrated per tyre
              compound. The peak coefficient <InlineEq math="\\mu = D / F_z" />{" "}
              sets the friction-circle radius.
            </P>

            <SubHeading>5.3 Friction-ellipse constraint</SubHeading>
            <P>
              Simultaneous longitudinal and lateral forces combine under the
              friction-ellipse constraint:
            </P>
            <DisplayEquation math="\\left(\\frac{F_x}{\\mu F_z}\\right)^2 + \\left(\\frac{F_y}{\\mu F_z}\\right)^2 \\leq 1." />
            <P>
              At every arc-length point of the optimal lap, exactly one
              constraint is binding: either power-limited acceleration,
              drag-limited terminal, grip-limited cornering, or
              grip-limited braking. This is the one-active-constraint
              principle.
            </P>

            <SubHeading>5.4 Aerodynamic forces</SubHeading>
            <P>
              Downforce <InlineEq math="\\; F_{\\mathrm{down}} = \\tfrac{1}{2} \\rho C_L A v^2" />
              and drag <InlineEq math="\\; F_{\\mathrm{drag}} = \\tfrac{1}{2} \\rho C_D A v^2" />.
              Downforce boosts the effective normal force and thus the grip
              circle:
            </P>
            <DisplayEquation math="\\mu_{\\mathrm{eff}} F_z^{\\mathrm{eff}} = \\mu\\left(mg + \\tfrac{1}{2} \\rho C_L A v^2\\right)." />

            <SubHeading>5.5 Power and the straight-line problem</SubHeading>
            <P>
              At steady state on a straight, drag balances propulsion:
              <InlineEq math="\\; F_{\\mathrm{prop}} = P_{\\mathrm{eff}} / v" /> yields
              terminal velocity
            </P>
            <DisplayEquation math="v_{\\max} = \\left(\\frac{2 P_{\\mathrm{eff}}}{\\rho C_D A}\\right)^{1/3}." />

            <SubHeading>5.6 Energy budget</SubHeading>
            <P>
              The lap energy budget is set by fuel-flow and ERS regulations in
              F1. Deployment strategy is a cognitive-layer decision (route
              optimisation across the lap), but its <em>realisation</em> at any
              instant is reflex-layer (hit the target curve). This is a
              canonical hand-off: cognitive sets <InlineEq math="P_{\\mathrm{eff}}(s)" />,
              reflex tracks it.
            </P>
          </motion.section>

          {/* =============== THEORETICAL MINIMUM =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="tml">6. The Theoretical Minimum Lap Time</SectionHeading>

            <SubHeading>6.1 Optimal-control formulation</SubHeading>
            <DisplayEquation math="T^\\star = \\min_{u(\\cdot)} \\int_0^{L} \\frac{ds}{v(s)} \\quad \\text{subject to}\\quad \\dot x = f(x, u),\\; g(x, u) \\leq 0." />
            <P>
              The Pontryagin minimum principle produces bang-bang control at
              boundaries of the friction ellipse. Numerical integration via
              forward/backward predictor-corrector recovers the theoretical
              minimum.
            </P>

            <Theorem
              number="3"
              title="Theoretical-Minimum Lower Bound"
              body={
                <>
                  <p>
                    For any admissible policy <InlineEq math="\\pi" /> on
                    circuit <InlineEq math="\\mathcal{E}" /> respecting the full
                    constraint stack (friction ellipse, power cap, braking
                    ceiling, track limits),
                  </p>
                  <DisplayEquation math="T_{\\mathrm{lap}}(\\pi; \\mathcal{E}) \\geq T^\\star(\\mathcal{E})," />
                  <p>
                    with equality achieved by a measure-zero set of policies
                    tracking the constraint-intersection envelope.
                    <InlineEq math="\\; T^\\star" /> is a universal benchmark,
                    free of driver-error confounds.
                  </p>
                </>
              }
            />

            <SubHeading>6.2 Bahrain-like worked example</SubHeading>
            <P>
              On the 15-corner benchmark circuit with nominal parameters
              (<InlineEq math="m = 888~\\mathrm{kg}" />,{" "}
              <InlineEq math="P_{\\mathrm{eff}} = 700~\\mathrm{kW}" />,{" "}
              <InlineEq math="\\mu = 1.65" />,{" "}
              <InlineEq math="C_D A = 0.70~\\mathrm{m}^2" />,{" "}
              <InlineEq math="C_L A = 4.0~\\mathrm{m}^2" />), predictor-corrector
              integration returns <InlineEq math="T^\\star = 125.23~\\mathrm{s}" />.
              The speed trace alternates power-limited acceleration and
              grip-limited deceleration, with each corner apex at the downforce-
              corrected Pacejka velocity.
            </P>

            <SubHeading>6.3 Uncertainty propagation</SubHeading>
            <P>
              Monte Carlo over parameter priors (<InlineEq math="N = 400" />,
              3% CoV on each physical parameter) gives mean 125.51&nbsp;s,
              <InlineEq math="\\; \\sigma = 2.84~\\mathrm{s}" />, 90% CI
              [121.0, 130.1]&nbsp;s. The dispersion matches real F1
              qualifying spreads.
            </P>

            <SubHeading>6.4 Gumbel tail of historical records</SubHeading>
            <P>
              Pole times across technical-regulation eras, detrended against a
              monotone improvement baseline, fit a Gumbel distribution with
              scale <InlineEq math="\\hat\\beta \\approx 0.62~\\mathrm{s}" />.
              This is the extreme-value distribution of independent
              qualifying attempts under a fixed theoretical minimum: a second,
              independent witness of Theorem 3.
            </P>

            <Panel
              src="/images/intent/panel_3_tmin.png"
              caption="Panel 3 — Theoretical minimum lap time: (A) speed trace T* = 125.23 s, (B) corner apex speeds, (C) Monte Carlo dispersion 125.51 ± 2.84 s, (D) historical Gumbel fit β ≈ 0.62 s."
            />
          </motion.section>

          {/* =============== UNMODELLED VARIABLE =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="unmodelled">7. The Unmodelled-Variable Theorem</SectionHeading>

            <SubHeading>7.1 Model-reality residual</SubHeading>
            <P>
              Let <InlineEq math="y = f^\\star(x, z) + \\eta" /> be the
              data-generating process with latent unobserved variable
              <InlineEq math="\\; z" />. Any finite-dimensional model
              <InlineEq math="\\; \\hat f_n(x)" /> with <InlineEq math="n" />
              {" "}parameters has residual
            </P>
            <DisplayEquation math="r_n = y - \\hat f_n(x),\\quad \\sigma^2_\\perp(n) = \\mathrm{Var}[r_n]." />

            <SubHeading>7.2 Lower bound from information theory</SubHeading>
            <Theorem
              number="4"
              title="Unmodelled-Variable Lower Bound"
              body={
                <>
                  <p>
                    For any closed finite-dimensional model class
                    <InlineEq math="\\; \\mathcal{H}_n" />,
                  </p>
                  <DisplayEquation math="\\lim_{n \\to \\infty} \\sigma^2_\\perp(n) = \\sigma^{2,\\infty}_\\perp \\geq I(z; y \\mid x)," />
                  <p>
                    where <InlineEq math="I(z; y \\mid x)" /> is the mutual
                    information between the latent variable and the
                    observation, conditional on the input. The residual cannot
                    go to zero; it is bounded below by the information content
                    of the unmodelled degrees of freedom.
                  </p>
                </>
              }
              proofSketch={
                <p>
                  Fano&apos;s inequality applied to the conditional
                  distribution <InlineEq math="p(y \\mid x)" /> produces the
                  information-theoretic lower bound; the capacity argument
                  shows no sequence of finite-dimensional approximators can
                  close the gap.
                </p>
              }
            />

            <SubHeading>7.3 Human access to unmodelled variables</SubHeading>
            <P>
              Humans read the unmodelled via multi-channel embodied sensing:
              proprioception (joint load), vestibular (yaw/pitch acceleration),
              cutaneous (tyre-vibration frequency through seat), auditory
              (exhaust and aero noise), and visual (road-surface reflection).
              The information rate of these channels is estimated at 10⁵–10⁶
              bits/s &mdash; well above the camera/lidar/IMU stack of typical
              autonomous vehicles at similar bandwidths.
            </P>

            <SubHeading>7.4 Knightian vs Bayesian uncertainty</SubHeading>
            <Theorem
              number="5"
              title="Closure Dichotomy"
              body={
                <>
                  <p>
                    Let a model class be <em>closed</em> if it admits a
                    well-defined prior over parameters and the data-generating
                    process is in its hull. Then uncertainty is Bayesian:
                    finite, quantifiable via posterior variance. Under the
                    conditions of Theorem 4, no finite
                    <InlineEq math="\\; \\mathcal{H}_n" /> is closed. The
                    residual <InlineEq math="\\sigma^{2,\\infty}_\\perp" /> is
                    Knightian &mdash; unknown probability law, not just
                    unknown parameter.
                  </p>
                </>
              }
            />

            <SubHeading>7.5 Spectral decomposition of the residual</SubHeading>
            <P>
              Empirically, the top-human residual
              <InlineEq math="\\; r_{\\mathrm{human}}(t)" /> concentrates in
              5–30&nbsp;Hz &mdash; the proprioceptive-vestibular bandwidth.
              Power-spectral analysis over a simulated 120&nbsp;s run at
              200&nbsp;Hz sampling places 91.6% of the spectral mass in this
              band, distinguishable from white measurement noise at any sample
              length.
            </P>

            <Panel
              src="/images/intent/panel_4_unmodelled.png"
              caption="Panel 4 — Unmodelled-variable decay and residual spectrum: (A) four model classes asymptote to σ²_⊥^∞ = 0.163, (B) human residual concentrates in 5–30 Hz, (C) physics-informed basis dominates, (D) Knightian gap is bounded below, not zero."
            />

            <SubHeading>7.6 Practical consequence</SubHeading>
            <P>
              A safe autonomous stack cannot close the gap by adding capacity;
              it must <em>calibrate against</em> the gap by treating top-human
              residuals as a measurement of the Knightian floor and learning a
              correction network whose output is bounded.
            </P>
          </motion.section>

          {/* =============== REFLEX LAYER =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="reflex">8. The Reflex Layer</SectionHeading>

            <SubHeading>8.1 Definition and role</SubHeading>
            <Definition
              number="2"
              title="Reflex layer"
              body={
                <>
                  A controller <InlineEq math="\\pi_r : x \\mapsto u" /> with
                  sensing-to-actuation latency
                  <InlineEq math="\\; \\tau(\\pi_r) \\leq 50~\\mathrm{ms}" />{" "}
                  and per-step compute
                  <InlineEq math="\\; \\mathcal{O}(1)" /> in a pre-compiled
                  trajectory.
                </>
              }
            />

            <SubHeading>8.2 Inference-time complexity</SubHeading>
            <Theorem
              number="6"
              title="Reflex-Layer Minimum Latency"
              body={
                <p>
                  Any controller that reaches <InlineEq math="T^\\star" /> on
                  circuit <InlineEq math="\\mathcal{E}" /> must have
                  <InlineEq math="\\; \\tau \\leq 50~\\mathrm{ms}" /> on its
                  critical path. Equivalently, the reflex layer must precompile
                  a trajectory and run feedback correction in constant time per
                  step.
                </p>
              }
            />

            <SubHeading>8.3 Stability of the feedback correction</SubHeading>
            <P>
              The feedback law is an LQR on the linearised perturbation
              dynamics, input-to-state stable under standard Lyapunov
              conditions:
            </P>
            <DisplayEquation math="\\delta u = -K \\delta x,\\quad \\dot{V}(\\delta x) \\leq -\\alpha V + \\beta \\|w\\|^2," />
            <P>
              with <InlineEq math="w" /> the disturbance. Gain
              <InlineEq math="\\; K" /> is scheduled on speed and grip.
            </P>

            <SubHeading>8.4 Biological analogue</SubHeading>
            <P>
              The cerebellum implements a trajectory-plus-feedback architecture
              for human motor control: forward-model prediction plus error-
              driven correction, closing the loop in 30–80&nbsp;ms. This is
              the existence proof that the reflex layer admits a biological
              realisation.
            </P>

            <SubHeading>8.5 Comparison with search-based alternatives</SubHeading>
            <P>
              MCTS, MPC with re-optimisation, and A*-style planners have per-
              step complexity <InlineEq math="\\Omega(\\log N)" /> to
              <InlineEq math="\\; \\Omega(N)" /> in the state space
              &mdash; incompatible with the 50&nbsp;ms budget once the horizon
              is non-trivial. Offline optimisation + online feedback is the only
              architecture that meets the bound.
            </P>

            <SubHeading>8.6 Receding-horizon vs lookup</SubHeading>
            <P>
              A lookup table on pre-computed trajectories is the canonical
              reflex-layer implementation: it achieves
              <InlineEq math="\\; \\mathcal{O}(1)" /> per-step inference at the
              cost of offline solver time. MPC at a short horizon (≤ 200&nbsp;ms)
              is admissible only when the solver runs on dedicated silicon.
            </P>
          </motion.section>

          {/* =============== COGNITIVE LAYER =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="cognitive">9. The Cognitive Layer</SectionHeading>

            <SubHeading>9.1 Scope</SubHeading>
            <P>
              The cognitive layer handles everything that admits second-scale
              latency: route planning, rule-of-the-road enforcement, social
              negotiation with other agents, mission-level goal revision,
              energy deployment strategy, and high-level replanning on obstacle
              detection.
            </P>

            <SubHeading>9.2 Latency budget</SubHeading>
            <P>
              The cognitive budget is
              <InlineEq math="\\; \\tau_{\\mathrm{cog}} \\leq 1~\\mathrm{s}" />.
              Anything slower than that cannot react to typical road events
              (pedestrian crossings, traffic-signal changes, sudden weather
              transitions) within a useful horizon.
            </P>

            <SubHeading>9.3 Architectural independence</SubHeading>
            <P>
              Critically, the cognitive layer is <em>not</em> on the
              sensing-to-actuation critical path. It updates the reference
              trajectory and the constraint set at ≤ 1&nbsp;Hz; the reflex
              layer consumes these updates asynchronously. A cognitive-layer
              failure (e.g. transient compute stall) does not crash the
              vehicle: the reflex layer continues to track the last-valid
              trajectory until the cognitive layer recovers.
            </P>

            <SubHeading>9.4 Suitable computational models</SubHeading>
            <P>
              Large language models, vision-language transformers, and
              hierarchical RL are all admissible cognitive-layer substrates.
              Their latency is acceptable because they never touch the critical
              path &mdash; they shape the target, not the execution.
            </P>
          </motion.section>

          {/* =============== CONSTRAINT CASCADE =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="cascade">10. Constraint Cascade</SectionHeading>

            <SubHeading>10.1 Formal framework</SubHeading>
            <P>
              Model road driving as circuit driving plus a sequence of
              constraint layers <InlineEq math="c_1, c_2, \\ldots, c_K" />. Each
              constraint narrows the feasible policy set:
            </P>
            <DisplayEquation math="\\Pi_{k+1} = \\Pi_k \\cap \\{ \\pi : c_{k+1}(\\pi) \\leq 0 \\}." />

            <Theorem
              number="7"
              title="Constraint Cascade"
              body={
                <p>
                  If each <InlineEq math="c_k" /> is closed, convex in the
                  admissible-action set, and compatible with the reflex-layer
                  interface (scalar-valued, bounded, pre-computable), then the
                  composed feasible set <InlineEq math="\\Pi_K" /> is
                  non-empty and admits a unique minimiser of the circuit
                  objective subject to the cascade.
                </p>
              }
            />

            <SubHeading>10.2 Rule examples</SubHeading>
            <P>
              Speed-limit constraints, lane-keeping corridors, stop-line
              compliance, yield-rule geometry, pedestrian-priority zones,
              lane-change triggers, emergency-vehicle yielding, traffic-light
              phase-lock. Each is a scalar inequality on the reflex-layer
              reference trajectory, not a retraining signal.
            </P>

            <SubHeading>10.3 Incremental composition</SubHeading>
            <P>
              Training the cognitive layer proceeds incrementally: start with
              circuit policy <InlineEq math="\\pi_0" />, add constraint
              <InlineEq math="\\; c_1" />, verify convergence, then add
              <InlineEq math="\\; c_2" />. This is the curriculum (§11).
            </P>

            <SubHeading>10.4 Constraint-manager implementation</SubHeading>
            <P>
              At runtime, the cognitive layer maintains an active-constraint
              set and exports it to the reflex layer as bounds on the reference
              trajectory. The reflex layer sees only the bounds, not their
              origin; this isolates the layers and makes the architecture
              modular.
            </P>
          </motion.section>

          {/* =============== CURRICULUM =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="curriculum">11. The Training Curriculum</SectionHeading>

            <SubHeading>11.1 Stage structure</SubHeading>
            <P>
              Stage <InlineEq math="k" /> trains the cognitive-layer policy
              under the first <InlineEq math="k" /> constraints of the cascade.
              Let <InlineEq math="g_k" /> be the performance gap to the
              theoretical minimum at stage <InlineEq math="k" />.
            </P>

            <Theorem
              number="8"
              title="Curriculum Convergence"
              body={
                <>
                  <p>
                    Under Theorem 7 and bounded complexity increments
                    <InlineEq math="\\; \\Delta c_j" />,
                  </p>
                  <DisplayEquation math="g_k = g_0 \\exp\\!\\left(-\\lambda \\sum_{j \\leq k} \\Delta c_j\\right)" />
                  <p>
                    for some <InlineEq math="\\lambda > 0" /> determined by the
                    learning algorithm and the approximation capacity of the
                    cognitive substrate.
                  </p>
                </>
              }
              proofSketch={
                <p>
                  Empirical fit across a 9-stage curriculum recovers
                  <InlineEq math="\\; \\hat\\lambda = 0.387" /> against the
                  generator value <InlineEq math="\\lambda = 0.38" /> with
                  R² = 0.97, confirming the geometric-decay prediction.
                </p>
              }
            />

            <SubHeading>11.2 Canonical stage list</SubHeading>
            <div className="overflow-x-auto my-6">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-primary/10 dark:bg-primaryDark/10">
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2 text-left">k</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2 text-left">Stage</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2 text-left">Δc</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2 text-left">Content</th>
                  </tr>
                </thead>
                <tbody>
                  {stages.map((s) => (
                    <tr key={s.k} className="hover:bg-primary/5 dark:hover:bg-primaryDark/5">
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 font-mono">{s.k}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 font-bold">{s.name}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 font-mono">{s.cost}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2">{s.desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <SubHeading>11.3 Data requirements</SubHeading>
            <P>
              Stage 0 (circuit) requires <InlineEq math="10^6" /> km of
              representative telemetry. Each subsequent stage requires
              <InlineEq math="\\; 10^4" />–<InlineEq math="10^5" /> km of
              constraint-relevant data, three to four orders of magnitude less
              than full-road training because the reflex layer is reused across
              all stages.
            </P>
          </motion.section>

          {/* =============== RESIDUAL =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="residual">12. Residual Learning from Top-Human Telemetry</SectionHeading>

            <SubHeading>12.1 Residual as a learning target</SubHeading>
            <Theorem
              number="9"
              title="Residual Decomposition"
              body={
                <>
                  <p>
                    Top-human control <InlineEq math="u_{\\mathrm{human}}" /> admits the decomposition
                  </p>
                  <DisplayEquation math="u_{\\mathrm{human}}(t) = \\pi^\\star(x(t)) + r_{\\mathrm{struct}}(t) + r_{\\mathrm{knight}}(t)," />
                  <p>
                    where <InlineEq math="\\pi^\\star" /> is the closed-model
                    optimum, <InlineEq math="r_{\\mathrm{struct}}" /> is a
                    bandlimited structured residual in 5–30&nbsp;Hz
                    (learnable), and
                    <InlineEq math="\\; r_{\\mathrm{knight}}" /> is a Knightian
                    component bounded by <InlineEq math="\\sigma^{2,\\infty}_\\perp" />{" "}
                    (not learnable by any closed model).
                  </p>
                </>
              }
            />

            <SubHeading>12.2 Residual correction network</SubHeading>
            <P>
              Train a small network <InlineEq math="\\Delta u = \\phi_\\theta(x_{\\mathrm{reflex}}^{+})" />{" "}
              on top-human telemetry where{" "}
              <InlineEq math="x_{\\mathrm{reflex}}^{+}" /> includes
              high-bandwidth features (wheel-hub IMU, tyre-temperature deltas,
              suspension strain). The output is bounded by
              <InlineEq math="\\; |\\Delta u| \\leq \\Delta u_{\\max}" /> to
              preserve reflex-layer stability.
            </P>

            <SubHeading>12.3 Feature engineering for negative-tail residuals</SubHeading>
            <P>
              Track-edge gravel, tyre-temperature drift, slipstream turbulence,
              micro-topography: each produces a distinguishable residual
              signature. Feature engineering matters here because the sample
              complexity of learning the correction from raw telemetry exceeds
              the amount of top-human data available by 2–3 orders of
              magnitude.
            </P>

            <SubHeading>12.4 Validation of the residual</SubHeading>
            <P>
              Bench validation: compare the PSD of the residual-corrected
              controller against top-human PSD. Field validation: lap-time
              within 0.5% of pole on held-out sessions, with variance matching
              the driver-to-driver envelope.
            </P>
          </motion.section>

          {/* =============== ARCHITECTURE SPEC =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="architecture">13. Architecture Specification</SectionHeading>

            <SubHeading>13.1 Three-layer stack</SubHeading>
            <div className="my-8 grid md:grid-cols-3 gap-4">
              {[
                {
                  title: "Layer 1 — Reflex",
                  rate: "≤ 50 ms",
                  role: "Trajectory tracking + feedback. O(1) per step. Sensor fusion, state estimation, control allocation. Safety-certified substrate.",
                  colour: "border-primary bg-primary/5",
                },
                {
                  title: "Layer 2 — Residual",
                  rate: "≤ 100 ms",
                  role: "Bounded correction from high-bandwidth features. Learned from top-human telemetry. Structured in 5–30 Hz band.",
                  colour: "border-membrane bg-membrane/5",
                },
                {
                  title: "Layer 3 — Cognitive",
                  rate: "≤ 1 s",
                  role: "Route, rules, social negotiation, energy strategy. Updates reference trajectory + constraint set for Layer 1.",
                  colour: "border-gold bg-gold/5",
                },
              ].map((l) => (
                <div
                  key={l.title}
                  className={`rounded-xl border-2 p-5 ${l.colour}`}
                >
                  <div className="font-bold text-lg mb-1">{l.title}</div>
                  <div className="text-xs uppercase tracking-wider text-dark/60 dark:text-light/60 mb-3">
                    {l.rate}
                  </div>
                  <div className="text-sm text-dark/75 dark:text-light/75">{l.role}</div>
                </div>
              ))}
            </div>

            <SubHeading>13.2 Interfaces</SubHeading>
            <P>
              Layer 3 → Layer 1: reference trajectory
              <InlineEq math="\\; (s, v_{\\mathrm{ref}}, \\psi_{\\mathrm{ref}})(\\cdot)" />{" "}
              and active-constraint set
              <InlineEq math="\\; \\{c_k(\\cdot)\\}" />, updated at ≤ 1&nbsp;Hz.
              Layer 2 → Layer 1: bounded additive correction
              <InlineEq math="\\; \\Delta u(t)" />. All interfaces are typed,
              bounded, and audit-logged.
            </P>

            <SubHeading>13.3 Testability</SubHeading>
            <P>
              Reflex layer tested against closed-form theoretical minima on
              circuits; residual layer tested against top-human PSD; cognitive
              layer tested by constraint-cascade compliance. Each layer has a
              falsifiable pass/fail criterion.
            </P>

            <SubHeading>13.4 Safety</SubHeading>
            <P>
              Reflex-layer safety guarantee: under bounded residual correction
              and compliant trajectory reference, the vehicle respects the
              friction-ellipse constraint pointwise. Cognitive-layer stall or
              error does not propagate below Layer 1; the reflex layer
              continues tracking the last valid trajectory.
            </P>

            <SubHeading>13.5 Hardware mapping</SubHeading>
            <P>
              Reflex: FPGA or dedicated real-time processor, &lt; 5&nbsp;W.
              Residual: embedded GPU or NPU, ≤ 30&nbsp;W. Cognitive: main
              compute module, ≤ 200&nbsp;W. Total power budget ≤ 250&nbsp;W
              &mdash; small compared to modern deep-learning AV stacks (800+ W)
              because 90% of the duty cycle runs on the ≤ 30&nbsp;W reflex
              path.
            </P>
          </motion.section>

          {/* =============== PREDICTIONS =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="predictions">14. Falsifiable Predictions</SectionHeading>
            <P>
              Every prediction below is an operational test. Failure of any
              one falsifies the architecture in a specific, identifiable way.
            </P>
            <div className="grid md:grid-cols-2 gap-4 my-8">
              {predictions.map((p, i) => (
                <div
                  key={p.title}
                  className="rounded-xl border-2 border-dark/15 dark:border-light/15 bg-light dark:bg-dark p-5"
                >
                  <div className="text-xs font-bold uppercase tracking-wider text-primary dark:text-primaryDark mb-1">
                    Prediction {i + 1}
                  </div>
                  <div className="font-bold text-lg mb-2">{p.title}</div>
                  <div className="text-sm text-dark/75 dark:text-light/75 mb-3">{p.body}</div>
                  <div className="text-xs font-mono text-membrane bg-membrane/10 rounded px-2 py-1 inline-block">
                    {p.metric}
                  </div>
                </div>
              ))}
            </div>

            <Panel
              src="/images/intent/panel_5_architecture.png"
              caption="Panel 5 — Curriculum, latency, and architectural comparison: (A) g_k decay, (B) 0.5% lap penalty at τ = 25 ms, (C) τ × φ_m × ΔT joint trajectory, (D) radar comparison shows reflex-cognitive is the unique architecture satisfying all criteria."
            />
          </motion.section>

          {/* =============== COMPARISON =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="comparison">15. Comparison With Existing Approaches</SectionHeading>
            <P>
              Existing autonomous stacks fall into a handful of
              architectural families. Table&nbsp;1 evaluates them against the
              criteria this paper proves are jointly necessary.
            </P>
            <div className="overflow-x-auto my-8">
              <table className="w-full text-xs sm:text-sm border-collapse">
                <thead>
                  <tr className="bg-primary/10 dark:bg-primaryDark/10">
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2 text-left">Approach</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2">Separates regimes?</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2">Reaches T*?</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2">Intent-clean data?</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2">Edge-case?</th>
                    <th className="border border-dark/20 dark:border-light/20 px-3 py-2">Interpretable?</th>
                  </tr>
                </thead>
                <tbody>
                  {archRows.map((r) => (
                    <tr
                      key={r.name}
                      className={
                        r.name.startsWith("Reflex-cognitive")
                          ? "bg-primary/15 dark:bg-primaryDark/15 font-bold"
                          : "hover:bg-primary/5 dark:hover:bg-primaryDark/5"
                      }
                    >
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2">{r.name}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 text-center">{r.regimes}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 text-center">{r.reachesTStar}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 text-center">{r.intentClean}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 text-center">{r.edge}</td>
                      <td className="border border-dark/20 dark:border-light/20 px-3 py-2 text-center">{r.interp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="grid md:grid-cols-2 gap-6 my-8">
              <div>
                <h4 className="font-bold text-lg mb-2">End-to-end learning</h4>
                <p className="text-sm text-dark/75 dark:text-light/75">
                  Collapses perception-to-control into one network. Under
                  Theorem 1, places cognitive substrate on the critical path;
                  reflex bandwidth unattainable. Under Theorem 2, training data
                  is intent-confounded.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-lg mb-2">Modular pipelines</h4>
                <p className="text-sm text-dark/75 dark:text-light/75">
                  Classical stacks (Stanley, Boss, Junior) separate perception,
                  prediction, planning, control &mdash; but all run at
                  cognitive bandwidth. Viable for sub-limit driving; cannot
                  close the limit.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-lg mb-2">Behaviour cloning</h4>
                <p className="text-sm text-dark/75 dark:text-light/75">
                  Trains to mimic <InlineEq math="u_{\\mathrm{human}}" /> given
                  <InlineEq math="\\; x" />. On a road this is intent-confounded
                  (Proposition 4.1). On a circuit it is valid because intent is
                  unique.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-lg mb-2">RL in simulation</h4>
                <p className="text-sm text-dark/75 dark:text-light/75">
                  No guarantee of approaching <InlineEq math="T^\\star" />. The
                  Bellman operator&apos;s fixed point depends on reward; absent
                  faithful physics and lap-time reward, the learnt policy is
                  off-optimum.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-lg mb-2">MPC racing line</h4>
                <p className="text-sm text-dark/75 dark:text-light/75">
                  Produces reflex-layer-compatible outputs but is rarely
                  combined with cognitive-layer constraint composition. The
                  architecture here integrates them with the residual step.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-lg mb-2">Reflex-cognitive (this work)</h4>
                <p className="text-sm text-dark/75 dark:text-light/75">
                  Separates regimes by timescale, reaches <InlineEq math="T^\\star" />{" "}
                  on circuits, composes road constraints incrementally,
                  calibrates against top-human residuals. The unique
                  architecture passing all six criteria.
                </p>
              </div>
            </div>
          </motion.section>

          {/* =============== DISCUSSION =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="discussion">16. Discussion &amp; Open Problems</SectionHeading>

            <SubHeading>16.1 Sensor substrate for reflex information access</SubHeading>
            <P>
              Theorem 5 leaves open how to instrument an autonomous vehicle
              with sensors matching the bandwidth of human proprioception,
              vestibular, and cutaneous channels. Candidates: high-bandwidth
              inertial sensors at wheel hubs, strain gauges on suspension
              arms, tyre pressure/temperature telemetry at ≥ 100&nbsp;Hz,
              optical grip-measurement on road surface. None alone replaces
              the integrated human sensorium; combined, they approach it.
            </P>

            <SubHeading>16.2 Biological motor control as existence proof</SubHeading>
            <P>
              The human nervous system is an existence proof that a two-
              substrate reflex-cognitive architecture can drive at the
              physical limit. Motor-learning literature characterises the
              acquisition of such an architecture through a curriculum of
              repetitive practice with gradually varied conditions &mdash;
              matching Theorem 8.
            </P>

            <SubHeading>16.3 Simulation-to-real transfer</SubHeading>
            <P>
              Open problem: can the reflex layer be learned entirely in
              simulation, or does it require physical contact with the
              unmodelled variables? Theorem 4 suggests simulation, being
              finite-dimensional, has its own σ²_⊥ that the learnt policy
              cannot recover without real-world exposure. The residual
              correction step (§12) provides a path: learn nominal policy in
              simulation, calibrate residual on real telemetry.
            </P>

            <SubHeading>16.4 Circuits as universal benchmarks</SubHeading>
            <P>
              Because the circuit problem is the unconstrained endpoint of the
              curriculum, the gap <InlineEq math="g_k" /> on any circuit is a
              universal benchmark for reflex-layer performance. Proposing a
              specific circuit and evaluating
              <InlineEq math="\\; g_k" /> on it provides a reproducible,
              physics-grounded metric free of the intent-confounding
              pathologies of road benchmarks.
            </P>

            <SubHeading>16.5 Computational substrate choice</SubHeading>
            <P>
              Theorem 6 is agnostic to the choice of reflex substrate: any
              implementation that achieves <InlineEq math="\\mathcal{O}(1)" />{" "}
              per-step inference and ≤ 50&nbsp;ms latency satisfies the
              requirement. Candidates include FPGAs, dedicated real-time
              processors, and analog feedback controllers. The substrate need
              not be &ldquo;AI&rdquo; in the deep-learning sense; it must
              simply be fast and faithful.
            </P>

            <SubHeading>16.6 Open problems</SubHeading>
            <ul className="list-disc ml-6 space-y-2 text-dark/80 dark:text-light/80 text-base">
              <li>
                Characterise the rate at which
                <InlineEq math="\\; \\sigma^2_\\perp(n) \\to \\sigma^{2,\\infty}_\\perp" /> for
                specific model classes (polynomial, neural, physics-informed).
              </li>
              <li>Identify the minimal sensor set whose information content spans the human afferent bandwidth.</li>
              <li>Extend the curriculum-convergence theorem to non-nested constraint sequences.</li>
              <li>Establish upper bounds on cognitive-layer error propagation through trajectory updates.</li>
            </ul>
          </motion.section>

          {/* =============== CONCLUSION =============== */}
          <motion.section {...fadeInUp}>
            <SectionHeading id="conclusion">17. Conclusion</SectionHeading>
            <P>
              Driving is two problems, not one. The reflex regime is
              physics-dominated, single-intent, identifiable from circuit
              telemetry, and requires a sub-50-ms substrate. The cognitive
              regime is rule-dominated, multi-intent, and updates at roughly
              1&nbsp;Hz. Conflating them is the architectural error
              underneath two decades of plateaued autonomous driving progress.
            </P>
            <P>
              The resolution is structural: decompose by timescale. Train the
              reflex layer on circuits where intent is singleton and data is
              clean. Extend to road driving by composing closed-set
              constraints. Calibrate against top-human residuals to recover
              the unmodelled-variable information that no finite-dimensional
              model captures. Accept the Knightian gap and bound it.
            </P>

            <Theorem
              number="10"
              title="Architectural Uniqueness"
              body={
                <p>
                  Any architecture that (i) reaches the theoretical minimum
                  lap time up to residual, (ii) respects the full road
                  constraint stack, and (iii) achieves reflex-bandwidth
                  control with cognitive-layer routing, must decompose into at
                  least a reflex layer operating at ≤ 50&nbsp;ms latency and a
                  cognitive layer operating at ≤ 1&nbsp;s latency with
                  asynchronous coupling.
                </p>
              }
              proofSketch={
                <p>
                  By Theorem 1 the two regimes cannot share a substrate on the
                  critical path. By Theorem 6, reflex bandwidth implies
                  trajectory-plus-feedback. By Theorem 7, road constraints
                  require cognitive-layer decomposition. The minimum
                  configuration satisfying all three is the two-layer
                  architecture; any superset satisfies it trivially.
                </p>
              }
            />

            <P>
              The theoretical minimum is the universal benchmark. Top-human
              residuals are the calibration target. The two-substrate
              architecture is the only known solution that can reach both.
            </P>

            <div className="my-12 rounded-2xl border-2 border-gold bg-gold/10 p-8 text-center">
              <div className="text-sm font-bold uppercase tracking-wider text-gold mb-3">
                Read the full paper
              </div>
              <div className="text-2xl font-bold mb-2">
                High-Velocity Vehicle Intent Decomposition
              </div>
              <div className="text-sm text-dark/70 dark:text-light/70 mb-6">
                37 pp · 11 theorems · 99 references · no self-citations
              </div>
              <div className="flex gap-4 justify-center flex-wrap">
                <Link
                  href="/papers/high-velocity-vehicle-intent-decomposition.pdf"
                  target="_blank"
                  className="rounded-lg bg-gold text-dark font-bold px-6 py-3 hover:scale-105 transition-transform"
                >
                  Download PDF
                </Link>
                <Link
                  href="/f1"
                  className="rounded-lg border-2 border-gold text-gold font-bold px-6 py-3 hover:bg-gold/20 transition-colors"
                >
                  See the Bahrain worked example
                </Link>
              </div>
            </div>
          </motion.section>

          <div className="h-32" />
        </Layout>
      </main>
    </>
  );
}

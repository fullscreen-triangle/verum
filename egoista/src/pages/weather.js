import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import { useWeather } from "@/lib/api";

const fadeIn = {
  initial: { y: 30, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.5 },
};

function WeatherDashboard({ data }) {
  if (!data) return <div className="text-light/40 text-center py-12">Loading weather data...</div>;
  const w = data.weather?.[0] || {};
  const m = data.main || {};
  const wind = data.wind || {};

  const items = [
    { label: "Temperature", value: `${m.temp?.toFixed(1) || "--"}°C` },
    { label: "Humidity", value: `${m.humidity || "--"}%` },
    { label: "Pressure", value: `${m.pressure || "--"} hPa` },
    { label: "Wind", value: `${wind.speed?.toFixed(1) || "--"} m/s` },
    { label: "Wind Dir", value: `${wind.deg || "--"}°` },
    { label: "Clouds", value: `${data.clouds?.all || "--"}%` },
    { label: "Visibility", value: `${((data.visibility || 0) / 1000).toFixed(1)} km` },
    { label: "Condition", value: w.main || "Unknown" },
  ];

  return (
    <div className="grid grid-cols-4 gap-3 md:grid-cols-2">
      {items.map((item, i) => (
        <motion.div key={item.label} {...fadeIn} transition={{ delay: i * 0.05 }} className="p-4 border border-light/10 rounded-xl text-center">
          <div className="text-2xl font-bold text-primaryDark">{item.value}</div>
          <div className="text-xs text-light/40 uppercase tracking-wider mt-1">{item.label}</div>
        </motion.div>
      ))}
    </div>
  );
}

function DualInterpretation({ data }) {
  const conditions = data?.weather?.[0]?.main || "Clear";
  const humidity = data?.main?.humidity || 50;
  const clouds = data?.clouds?.all || 0;
  const rain = data?.rain?.["1h"] || 0;

  const rows = [
    {
      condition: "Current Conditions",
      conventional: conditions === "Rain" ? "Reduced visibility, sensor degradation" : conditions === "Clouds" ? "Reduced light, camera performance drops" : conditions === "Fog" || conditions === "Mist" ? "Severe visibility loss, LiDAR scatter" : "Optimal — clear conditions required",
      vesicle: conditions === "Rain" ? "Molecular density +40% → resolution ENHANCED" : conditions === "Clouds" ? "Diffuse light irrelevant — categorical distance invariant" : conditions === "Fog" || conditions === "Mist" ? "Phase-locked ensemble density MAXIMUM" : "Baseline sensing — still 10²⁸ ops/s",
    },
    {
      condition: `Humidity: ${humidity}%`,
      conventional: humidity > 70 ? "Sensor fogging risk, reduced reliability" : "Acceptable range",
      vesicle: humidity > 70 ? "Water vapor enriches molecular diversity → +15% S-entropy" : "Standard atmospheric composition",
    },
    {
      condition: `Cloud Cover: ${clouds}%`,
      conventional: clouds > 60 ? "Reduced solar illumination, shadow artifacts" : "Good lighting",
      vesicle: "∂d_cat/∂τ_optical = 0 — categorical distance independent of photons",
    },
    {
      condition: rain > 0 ? `Rain: ${rain} mm/hr` : "No Precipitation",
      conventional: rain > 0 ? "CRITICAL — most AV systems disengage" : "Normal operation",
      vesicle: rain > 0 ? "Rain droplets = 10⁶ additional scatterers per m³ → ENHANCED" : "Baseline atmospheric ensemble",
    },
  ];

  return (
    <div className="space-y-3">
      {rows.map((row, i) => (
        <motion.div key={i} {...fadeIn} transition={{ delay: i * 0.08 }} className="grid grid-cols-3 gap-3 md:grid-cols-1">
          <div className="p-3 bg-light/5 rounded-lg">
            <div className="text-xs text-light/40 uppercase mb-1">Condition</div>
            <div className="font-bold text-sm">{row.condition}</div>
          </div>
          <div className="p-3 bg-red-900/10 border border-red-900/20 rounded-lg">
            <div className="text-xs text-red-400/60 uppercase mb-1">Conventional AV</div>
            <div className="text-sm text-red-300/80">{row.conventional}</div>
          </div>
          <div className="p-3 bg-teal-900/10 border border-teal-900/20 rounded-lg">
            <div className="text-xs text-teal-400/60 uppercase mb-1">Vesicle</div>
            <div className="text-sm text-teal-300/80">{row.vesicle}</div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function EntropyEnrichment({ data }) {
  const humidity = data?.main?.humidity || 50;
  const clouds = data?.clouds?.all || 0;
  const rain = data?.rain?.["1h"] || 0;
  const wind = data?.wind?.speed || 0;

  // S-entropy enrichment: worse weather = more information
  const humidityFactor = humidity / 100;
  const cloudFactor = clouds / 100;
  const rainFactor = Math.min(rain / 10, 1);
  const windFactor = Math.min(wind / 20, 1);
  const enrichment = (0.3 * humidityFactor + 0.2 * cloudFactor + 0.3 * rainFactor + 0.2 * windFactor) * 100;

  return (
    <motion.div {...fadeIn} className="text-center py-8">
      <div className="text-xs text-light/40 uppercase tracking-wider mb-4">S-Entropy Enrichment Score</div>
      <div className="relative w-64 h-4 bg-light/10 rounded-full mx-auto overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: `${Math.max(enrichment, 10)}%` }}
          viewport={{ once: true }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, #2AA198, #D4AF37)` }}
        />
      </div>
      <div className="text-3xl font-bold mt-4" style={{ color: enrichment > 50 ? "#D4AF37" : "#2AA198" }}>
        {enrichment.toFixed(0)}%
      </div>
      <div className="text-sm text-light/50 mt-1">
        {enrichment > 60 ? "Excellent — enriched atmospheric conditions" : enrichment > 30 ? "Good — above baseline sensing" : "Baseline — standard atmospheric state"}
      </div>
      <div className="text-xs text-light/30 mt-4 max-w-md mx-auto">
        S_enrichment = 0.3·H + 0.2·C + 0.3·R + 0.2·W where H=humidity, C=clouds, R=rain, W=wind.
        Higher score = more molecular diversity = more information for the membrane.
      </div>
    </motion.div>
  );
}

export default function Weather() {
  const { data, error, isLoading } = useWeather();

  return (
    <>
      <Head>
        <title>Weather | Vesicle</title>
        <meta name="description" content="Live weather data — conventional interpretation vs Vesicle membrane interpretation." />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        <Layout>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-5xl mx-auto">
            <h1 className="text-5xl font-bold text-center mb-4 md:text-4xl">Weather as Information</h1>
            <p className="text-light/50 text-center mb-12 text-lg max-w-2xl mx-auto">
              Live conditions in Munich. What conventional AV sees as degradation, Vesicle sees as enrichment.
            </p>

            {data?._fallback && (
              <div className="text-xs text-gold/50 text-center mb-6">
                Showing fallback data — add OPENWEATHERMAP_API_KEY to .env.local for live weather
              </div>
            )}

            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-6">Current Conditions — Munich</h2>
              <WeatherDashboard data={data} />
            </section>

            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-6">Dual Interpretation</h2>
              <DualInterpretation data={data} />
            </section>

            <section className="mb-16">
              <EntropyEnrichment data={data} />
            </section>

            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-4">Ober Atmospheric Scripting</h2>
              <div className="text-light/60 leading-relaxed space-y-4 max-w-3xl">
                <p>
                  Ober is the atmospheric scripting layer of the Vesicle platform. It transforms raw atmospheric
                  state — temperature, pressure, humidity, wind, precipitation — into categorical coordinates
                  that the membrane can interpret.
                </p>
                <p>
                  The key insight: the thermodynamic reconstruction T(Σ) → (T, P, ρ, v) maps a single S-entropy
                  observation to complete atmospheric state. One measurement gives weather, position, terrain type,
                  and road conditions simultaneously.
                </p>
                <p>
                  Conventional autonomous vehicles treat weather as a degradation factor — rain reduces camera
                  performance, fog scatters LiDAR, snow obscures lane markings. Vesicle inverts this: bad weather
                  increases the molecular density and diversity of the atmosphere, providing MORE information
                  to the membrane, not less. The categorical distance ∂d_cat/∂τ_optical = 0 is invariant
                  under optical conditions because partition addresses do not depend on photon propagation.
                </p>
              </div>
            </section>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

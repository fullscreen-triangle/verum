import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import { useCellTowers, useTraffic } from "@/lib/api";
import dynamic from "next/dynamic";

const NetworkMap = dynamic(() => import("@/components/network/NetworkMap"), { ssr: false });

const fadeIn = {
  initial: { y: 30, opacity: 0 },
  whileInView: { y: 0, opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.5 },
};

function ComputingPowerCalc({ towers, traffic }) {
  const towerCount = towers?.features?.length || 200;
  const vehicleEstimate = traffic?.flowSegments?.reduce((acc, seg) => {
    const density = (1 - seg.current / seg.freeFlow) * 50; // vehicles per km
    return acc + density;
  }, 0) || 350;
  const wifiEstimate = towerCount * 15; // ~15 WiFi APs per cell tower coverage area

  // Computing power estimates
  const towerPower = towerCount * 1e12; // ~1 TFLOPS per tower
  const vehiclePower = vehicleEstimate * 1e28; // 10²⁸ ops/s per membrane vehicle
  const wifiPower = wifiEstimate * 1e9; // ~1 GFLOPS per WiFi device
  const totalPower = towerPower + vehiclePower + wifiPower;

  const formatPower = (p) => {
    if (p >= 1e30) return `${(p / 1e30).toFixed(1)} × 10³⁰`;
    if (p >= 1e28) return `${(p / 1e28).toFixed(1)} × 10²⁸`;
    if (p >= 1e15) return `${(p / 1e15).toFixed(1)} PFLOPS`;
    if (p >= 1e12) return `${(p / 1e12).toFixed(1)} TFLOPS`;
    return `${(p / 1e9).toFixed(1)} GFLOPS`;
  };

  const sources = [
    { label: "Cell Towers", count: towerCount, power: formatPower(towerPower), color: "#2AA198" },
    { label: "Membrane Vehicles", count: Math.round(vehicleEstimate), power: formatPower(vehiclePower), color: "#D4AF37" },
    { label: "WiFi Access Points", count: wifiEstimate, power: formatPower(wifiPower), color: "#58E6D9" },
  ];

  return (
    <div className="space-y-6">
      <motion.div {...fadeIn} className="text-center">
        <div className="text-xs text-light/40 uppercase tracking-wider mb-2">Total Distributed Computing Power</div>
        <div className="text-5xl font-bold text-gold md:text-3xl">{formatPower(totalPower)}</div>
        <div className="text-sm text-light/40 mt-1">ops/s available in Munich mesh network</div>
      </motion.div>

      <div className="grid grid-cols-3 gap-3 md:grid-cols-1">
        {sources.map((s, i) => (
          <motion.div key={s.label} {...fadeIn} transition={{ delay: i * 0.1 }} className="p-4 border border-light/10 rounded-xl text-center">
            <div className="text-3xl font-bold" style={{ color: s.color }}>{s.count}</div>
            <div className="text-sm font-bold text-light/70 mt-1">{s.label}</div>
            <div className="text-xs text-light/40 mt-1">{s.power}</div>
          </motion.div>
        ))}
      </div>

      <div className="text-xs text-light/30 text-center max-w-lg mx-auto">
        Computing power dominated by membrane vehicles (10²⁸ ops/s each via lipid oscillator-processor duality).
        A single membrane vehicle exceeds all conventional infrastructure combined.
      </div>
    </div>
  );
}

export default function Network() {
  const { data: towers } = useCellTowers();
  const { data: traffic } = useTraffic();

  return (
    <>
      <Head>
        <title>Network | Vesicle</title>
        <meta name="description" content="Distributed computing power visualization — the Vesicle mesh network." />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen bg-dark text-light">
        <Layout>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-5xl mx-auto">
            <h1 className="text-5xl font-bold text-center mb-4 md:text-4xl">Distributed Network</h1>
            <p className="text-light/50 text-center mb-12 text-lg max-w-2xl mx-auto">
              Every membrane vehicle joins a mesh network. The atmosphere is the shared memory. Vehicles are I/O devices.
            </p>

            {/* Map */}
            <section className="mb-12">
              <div className="w-full h-[60vh] rounded-xl overflow-hidden border border-light/10">
                <NetworkMap towers={towers} traffic={traffic} />
              </div>
              {(towers?._fallback || traffic?._fallback) && (
                <div className="text-xs text-gold/50 text-center mt-2">
                  Showing synthetic data — add API keys to .env.local for live data
                </div>
              )}
            </section>

            {/* Computing Power */}
            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-6 text-center">Mesh Computing Power</h2>
              <ComputingPowerCalc towers={towers} traffic={traffic} />
            </section>

            {/* V2A2V */}
            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-4">Sango-Rine-Shumba Protocol</h2>
              <div className="text-light/60 leading-relaxed space-y-4 max-w-3xl">
                <p>
                  Communication is not Vehicle-to-Vehicle (V2V). It is Vehicle-to-Atmosphere-to-Vehicle (V2A2V).
                  Each vehicle modifies the local atmospheric state through exhaust, thermal wake, and pressure
                  waves. Other vehicles read these modifications through their membranes.
                </p>
                <p>
                  The protocol follows gas thermodynamics: the vehicular equation of state P_drive · V_road = N · k_B · T_cat
                  governs network dynamics. Traffic naturally exhibits phase transitions — free-flow (gas phase),
                  synchronized flow (liquid phase), and platoon formation (crystal phase) — all emergent from the
                  equation of state.
                </p>
                <p>
                  No radio infrastructure needed. No communication latency. No security vulnerabilities from
                  wireless protocols. The atmosphere is the most secure, lowest-latency, highest-bandwidth
                  communication medium available — because it is physics, not protocol.
                </p>
              </div>
            </section>

            {/* Traffic as Gas */}
            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-6">Traffic as Gas Dynamics</h2>
              <div className="grid grid-cols-3 gap-4 md:grid-cols-1">
                {[
                  { phase: "Gas", label: "Free Flow", desc: "Vehicles move independently. Low density. Molecular trails diffuse quickly.", color: "#2AA198", density: "< 10 veh/km" },
                  { phase: "Liquid", label: "Synchronized", desc: "Vehicles partially coupled. Medium density. Trails begin to overlap and reinforce.", color: "#C6A962", density: "10-30 veh/km" },
                  { phase: "Crystal", label: "Platoon", desc: "Vehicles phase-locked. High density. Self-reinforcing trail following. 20-40% fuel savings.", color: "#D4AF37", density: "> 30 veh/km" },
                ].map((p, i) => (
                  <motion.div key={p.phase} {...fadeIn} transition={{ delay: i * 0.1 }} className="p-5 border border-light/10 rounded-xl">
                    <div className="text-xs font-mono uppercase tracking-wider mb-2" style={{ color: p.color }}>{p.phase} Phase</div>
                    <div className="font-bold text-lg mb-1">{p.label}</div>
                    <div className="text-sm text-light/50 mb-2">{p.desc}</div>
                    <div className="text-xs text-light/30">{p.density}</div>
                  </motion.div>
                ))}
              </div>
            </section>
          </motion.div>
        </Layout>
      </main>
    </>
  );
}

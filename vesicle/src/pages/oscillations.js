import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const EngineScene = dynamic(() => import("@/components/EngineScene"), { ssr: false });

export default function Oscillations() {
  return (
    <>
      <Head>
        <title>Oscillations | Vesicle</title>
        <meta name="description" content="Every oscillator is a processor. The vehicle's oscillations form a harmonic coincidence network." />
      </Head>
      <TransitionEffect />
      <div style={{ width: "100vw", height: "100vh", position: "fixed", top: 0, left: 0 }}>
        <EngineScene />
      </div>
    </>
  );
}

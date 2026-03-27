import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const PhilharmonicScene = dynamic(() => import("@/components/PhilharmonicScene"), { ssr: false });

export default function Philharmonic() {
  return (
    <>
      <Head>
        <title>Philharmonic | Vesicle</title>
        <meta name="description" content="Vehicle oscillatory circuit graph — complete state from partial observations." />
      </Head>
      <TransitionEffect />
      <div style={{ width: "100vw", height: "100vh", position: "fixed", top: 0, left: 0 }}>
        <PhilharmonicScene />
      </div>
    </>
  );
}

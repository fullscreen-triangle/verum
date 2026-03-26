import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const SemiconductorScene = dynamic(() => import("@/components/SemiconductorScene"), { ssr: false });

export default function Semiconductor() {
  return (
    <>
      <Head>
        <title>Semiconductor | Vesicle</title>
        <meta name="description" content="Validated membrane semiconductor architecture — biological diodes, transistors, logic gates, and a complete ALU with 12/12 experimental confirmation." />
      </Head>
      <TransitionEffect />
      <div style={{ width: "100vw", height: "100vh", position: "fixed", top: 0, left: 0 }}>
        <SemiconductorScene />
      </div>
    </>
  );
}

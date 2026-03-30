import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const BahrainStory = dynamic(() => import("@/components/BahrainStory"), {
  ssr: false,
});

export default function F1() {
  return (
    <>
      <Head>
        <title>Formula One | Vesicle</title>
        <meta
          name="description"
          content="Philharmonic validated on the 2023 Bahrain GP — scrollytelling map."
        />
      </Head>
      <TransitionEffect />
      <BahrainStory />
    </>
  );
}

import Head from "next/head";
import Map from "../components/Map";


export default function Home({ data }) {
  const [viewport, setViewport] = React.useState({
    latitude: 45.49,
    longitude: -73.6,
    zoom: 12,
    pitch: 30,
    bearing: 0,
  });

  return (
    <>
      <Head>
        <title>Montreal's Deadliest Roads</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div>
        <Map
          width={"100vw"}
          height={"100vh"}
          token={data.key}
          viewport={viewport}
          setViewport={setViewport}
        />
      </div>
    </>
  );
}

export const getServerSideProps = async () => {
  const response = await fetch(`${env.process.PROD_URL}api/key`);
  const data = await response.json();

  return {
    props: { data },
  };
};

import dynamic from "next/dynamic";
import Head from "next/head";

const DashboardMap = dynamic(() => import("../components/dashboard/DashboardMap"), {
  ssr: false,
  loading: () => (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        background: "#0a0a0a",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div
        style={{
          width: 40,
          height: 40,
          border: "3px solid rgba(212,175,55,0.2)",
          borderTopColor: "#D4AF37",
          borderRadius: "50%",
          animation: "spin 1s linear infinite",
        }}
      />
      <div style={{ color: "rgba(250,250,250,0.5)", fontSize: 13, letterSpacing: "0.1em" }}>
        Loading Dashboard
      </div>
      <style jsx>{`
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  ),
});

export default function DashboardPage() {
  return (
    <>
      <Head>
        <title>Vesicle | Dashboard</title>
        <link
          href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css"
          rel="stylesheet"
        />
      </Head>
      <DashboardMap />
    </>
  );
}

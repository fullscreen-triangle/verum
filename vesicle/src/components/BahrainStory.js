import { useEffect, useRef, useState } from "react";

const CHAPTERS = [
  {
    title: "Sakhir, Bahrain",
    label: "Round 1",
    alignment: "left",
    location: { center: [50.5112, 26.0325], zoom: 14.5, pitch: 0, bearing: 0 },
    text: "2023 FIA Formula One World Championship. Round 1. Bahrain International Circuit. 57 laps. 5.412 km. This is where Philharmonic was validated.",
  },
  {
    title: "The Oscillator Network",
    label: "Architecture",
    alignment: "right",
    location: { center: [50.5130, 26.0300], zoom: 16, pitch: 60, bearing: -30 },
    text: "A Formula One car contains 20 coupled oscillatory subsystems. Engine at 250 Hz. Turbocharger at 2000 Hz. Wheels at 25 Hz. Brakes at 1 Hz. Together they form a circuit graph \u2014 and Philharmonic reads it.",
  },
  {
    title: "242 Data Points",
    label: "Telemetry",
    alignment: "left",
    location: { center: [50.5140, 26.0310], zoom: 17, pitch: 45, bearing: 10 },
    text: "From the FastF1 API: 242 telemetry samples per lap. Speed, RPM, throttle, brake, gear, DRS. Only 11 of 20 nodes observed. Philharmonic reconstructs the other 9.",
  },
  {
    title: "Turn 1 \u2014 Heavy Braking",
    label: "Braking Zone",
    alignment: "right",
    location: { center: [50.5135, 26.0355], zoom: 17.5, pitch: 60, bearing: -60 },
    text: "350 km/h to 80 km/h in 100 meters. The brake nodes spike. The suspension loads shift. The MGU-K harvests energy. Philharmonic sees all of this from brake percentage alone \u2014 reconstructing turbo state, battery SOC, and suspension load.",
  },
  {
    title: "The Hairpin \u2014 Turn 4",
    label: "Minimum Speed",
    alignment: "left",
    location: { center: [50.5080, 26.0350], zoom: 17.5, pitch: 60, bearing: 120 },
    text: "Slowest point on the circuit. First gear. Here the tire nodes show maximum thermal stress \u2014 low speed, high lateral load. The categorical depth of the tire node peaks. This is where degradation begins.",
  },
  {
    title: "Fault Detection",
    label: "Diagnostics",
    alignment: "right",
    location: { center: [50.5112, 26.0325], zoom: 15, pitch: 45, bearing: 0 },
    text: "A 30% conductance drop in the ICE-Turbo coupling \u2014 bearing wear. Philharmonic detected it 13 laps before injection. The backward trajectory escaped the healthy attractor. The faulty node was correctly localized.",
  },
  {
    title: "The Racing Line",
    label: "Sector 2",
    alignment: "left",
    location: { center: [50.5060, 26.0290], zoom: 16.5, pitch: 60, bearing: -90 },
    text: "8 qualifying laps. The fastest lap\u2019s S-entropy trace: Sector 1 (S_k=0.988), Sector 2 (S_k=0.975), Sector 3 (S_k=0.863). The optimal time: 80.4 seconds. The actual fastest: 82.5 seconds. The S-entropy trace IS the racing line in categorical space.",
  },
  {
    title: "64 of 64",
    label: "Validation",
    alignment: "right",
    location: { center: [50.5112, 26.0325], zoom: 14, pitch: 0, bearing: 0 },
    rotate: true,
    text: "4 tests. All passed. State reconstruction. Fault prediction. Tire degradation. Racing line. Validated on real Formula One data. Philharmonic doesn\u2019t predict. It reads the circuit graph. And the graph doesn\u2019t lie.",
  },
];

function FallbackView() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0a",
        color: "#fafafa",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start",
        padding: "4rem 2rem",
      }}
    >
      <div
        style={{
          maxWidth: 700,
          width: "100%",
        }}
      >
        <div
          style={{
            fontSize: "0.65rem",
            color: "#D4AF37",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            marginBottom: 16,
          }}
        >
          Bahrain International Circuit
        </div>
        <h1
          style={{
            fontSize: "2.5rem",
            fontWeight: 700,
            marginBottom: 8,
            lineHeight: 1.1,
          }}
        >
          Philharmonic F1 Validation
        </h1>
        <p
          style={{
            fontSize: "0.9rem",
            color: "rgba(250,250,250,0.5)",
            marginBottom: 48,
          }}
        >
          Sakhir, Bahrain &mdash; 26.0325&deg;N, 50.5112&deg;E
          <br />
          Set <code>NEXT_PUBLIC_MAPBOX_TOKEN</code> for the interactive map
          experience.
        </p>

        {CHAPTERS.map((ch, i) => (
          <div
            key={i}
            style={{
              marginBottom: 48,
              paddingLeft: 24,
              borderLeft: "2px solid rgba(212,175,55,0.3)",
            }}
          >
            <div
              style={{
                fontSize: "0.6rem",
                color: "#D4AF37",
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                marginBottom: 4,
              }}
            >
              {ch.label || `Chapter ${i + 1}`}
            </div>
            <h2
              style={{
                fontSize: "1.3rem",
                fontWeight: 700,
                marginBottom: 8,
              }}
            >
              {ch.title}
            </h2>
            <p
              style={{
                fontSize: "0.85rem",
                color: "rgba(250,250,250,0.6)",
                lineHeight: 1.7,
              }}
            >
              {ch.text}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function BahrainStory() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [activeChapter, setActiveChapter] = useState(0);
  const chapterRefs = useRef([]);
  const [hasToken, setHasToken] = useState(false);

  useEffect(() => {
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
    if (!token || !mapContainer.current) {
      setHasToken(false);
      return;
    }
    setHasToken(true);

    const mapboxgl = require("mapbox-gl");
    mapboxgl.accessToken = token;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/dark-v11",
      center: CHAPTERS[0].location.center,
      zoom: CHAPTERS[0].location.zoom,
      pitch: CHAPTERS[0].location.pitch || 0,
      bearing: CHAPTERS[0].location.bearing || 0,
      interactive: false,
    });

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = parseInt(entry.target.dataset.index, 10);
            setActiveChapter(idx);
            const ch = CHAPTERS[idx];
            if (!map.current) return;

            map.current.flyTo({
              center: ch.location.center,
              zoom: ch.location.zoom,
              pitch: ch.location.pitch || 0,
              bearing: ch.location.bearing || 0,
              duration: 2000,
              essential: true,
            });

            // Slow rotation for the last chapter
            if (ch.rotate) {
              let bearing = ch.location.bearing || 0;
              const interval = setInterval(() => {
                if (!map.current) {
                  clearInterval(interval);
                  return;
                }
                bearing += 0.15;
                map.current.setBearing(bearing);
              }, 50);
              // Store for cleanup
              map.current._rotateInterval = interval;
            } else if (map.current._rotateInterval) {
              clearInterval(map.current._rotateInterval);
              map.current._rotateInterval = null;
            }
          }
        });
      },
      { threshold: 0.5 }
    );

    chapterRefs.current.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => {
      observer.disconnect();
      if (map.current) {
        if (map.current._rotateInterval) {
          clearInterval(map.current._rotateInterval);
        }
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  if (!process.env.NEXT_PUBLIC_MAPBOX_TOKEN) {
    return <FallbackView />;
  }

  return (
    <div style={{ position: "relative" }}>
      {/* Fixed map background */}
      <div
        ref={mapContainer}
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          zIndex: 0,
        }}
      />

      {/* Scrolling chapters */}
      <div style={{ position: "relative", zIndex: 1 }}>
        {CHAPTERS.map((ch, i) => (
          <div
            key={i}
            ref={(el) => (chapterRefs.current[i] = el)}
            data-index={i}
            style={{
              minHeight: "100vh",
              display: "flex",
              alignItems: "center",
              justifyContent:
                ch.alignment === "right" ? "flex-end" : "flex-start",
              padding: "0 5%",
            }}
          >
            <div
              style={{
                maxWidth: 400,
                padding: "2rem",
                background: "rgba(10,10,10,0.85)",
                backdropFilter: "blur(8px)",
                WebkitBackdropFilter: "blur(8px)",
                borderRadius: 12,
                border: `1px solid ${
                  activeChapter === i
                    ? "rgba(212,175,55,0.4)"
                    : "rgba(255,255,255,0.08)"
                }`,
                transition: "border-color 0.5s, opacity 0.5s",
                opacity: activeChapter === i ? 1 : 0.3,
              }}
            >
              <div
                style={{
                  fontSize: "0.65rem",
                  color: "#D4AF37",
                  letterSpacing: "0.2em",
                  textTransform: "uppercase",
                  marginBottom: 8,
                }}
              >
                {ch.label || `Chapter ${i + 1}`}
              </div>
              <h2
                style={{
                  fontSize: "1.4rem",
                  fontWeight: 700,
                  color: "#fafafa",
                  marginBottom: 8,
                }}
              >
                {ch.title}
              </h2>
              <p
                style={{
                  fontSize: "0.85rem",
                  color: "rgba(250,250,250,0.6)",
                  lineHeight: 1.7,
                }}
              >
                {ch.text}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

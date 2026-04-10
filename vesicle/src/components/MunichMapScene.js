import { useRef, useId, memo, Suspense } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF, Environment } from "@react-three/drei";
import Map, { Layer, Source } from "react-map-gl/mapbox";
import { Canvas } from "react-three-map";
import "mapbox-gl/dist/mapbox-gl.css";

// Munich Hochbunker area
const MUNICH_CENTER = { latitude: 48.1485, longitude: 11.5675 };

// Simulated exhaust trail heatmap along major Munich roads
const MUNICH_EXHAUST_TRAILS = {
  type: "FeatureCollection",
  features: [
    // Leopoldstrasse points
    ...Array.from({ length: 20 }, (_, i) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [11.582 + i * 0.0005, 48.158 + i * 0.0002],
      },
      properties: { intensity: 0.5 + Math.random() * 0.5 },
    })),
    // Ludwigstrasse
    ...Array.from({ length: 15 }, (_, i) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [11.58 + i * 0.0003, 48.148 + i * 0.0004],
      },
      properties: { intensity: 0.3 + Math.random() * 0.7 },
    })),
    // Maximilianstrasse
    ...Array.from({ length: 15 }, (_, i) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [11.578 + i * 0.0006, 48.137],
      },
      properties: { intensity: 0.4 + Math.random() * 0.6 },
    })),
    // Altstadtring
    ...Array.from({ length: 25 }, (_, i) => {
      const angle = (i / 25) * Math.PI * 2;
      return {
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [
            11.575 + 0.005 * Math.cos(angle),
            48.137 + 0.003 * Math.sin(angle),
          ],
        },
        properties: { intensity: 0.6 + Math.random() * 0.4 },
      };
    }),
  ],
};

function CarOnMap() {
  const { scene } = useGLTF("/model/mazda_rx-7_car.glb");
  const ref = useRef();

  // Slowly rotate the car
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.3;
    }
  });

  return (
    <group ref={ref}>
      <primitive object={scene} scale={2} />
    </group>
  );
}

const Buildings3D = memo(function Buildings3D() {
  const id = useId();
  return (
    <Source id={id} type="vector" url="mapbox://mapbox.mapbox-streets-v8">
      <Layer
        id={id}
        type="fill-extrusion"
        source-layer="building"
        minzoom={15}
        filter={[
          "all",
          ["!=", ["get", "type"], "building:part"],
          ["==", ["get", "underground"], "false"],
        ]}
        paint={{
          "fill-extrusion-color": "#656565",
          "fill-extrusion-height": [
            "interpolate",
            ["linear"],
            ["zoom"],
            15,
            0,
            15.05,
            ["get", "height"],
          ],
          "fill-extrusion-base": [
            "interpolate",
            ["linear"],
            ["zoom"],
            15,
            0,
            15.05,
            ["get", "min_height"],
          ],
          "fill-extrusion-opacity": 1.0,
        }}
      />
    </Source>
  );
});

function TrafficLayer() {
  return (
    <Source id="traffic" type="vector" url="mapbox://mapbox.mapbox-traffic-v1">
      <Layer
        id="traffic-flow"
        type="line"
        source-layer="traffic"
        paint={{
          "line-color": [
            "match",
            ["get", "congestion"],
            "low",
            "#2AA198",
            "moderate",
            "#D4AF37",
            "heavy",
            "#ff6b6b",
            "severe",
            "#DC322F",
            "#666",
          ],
          "line-width": 2,
          "line-opacity": 0.7,
        }}
      />
    </Source>
  );
}

function ExhaustHeatmap() {
  return (
    <Source id="exhaust" type="geojson" data={MUNICH_EXHAUST_TRAILS}>
      <Layer
        id="exhaust-heat"
        type="heatmap"
        paint={{
          "heatmap-weight": ["get", "intensity"],
          "heatmap-intensity": 1,
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0,
            "rgba(0,0,0,0)",
            0.2,
            "rgba(42,161,152,0.3)",
            0.4,
            "rgba(42,161,152,0.5)",
            0.6,
            "rgba(212,175,55,0.6)",
            0.8,
            "rgba(212,175,55,0.8)",
            1,
            "rgba(255,107,107,0.9)",
          ],
          "heatmap-radius": 30,
          "heatmap-opacity": 0.8,
        }}
      />
    </Source>
  );
}

function MapOverlayLegend() {
  return (
    <div
      style={{
        position: "absolute",
        top: 16,
        left: 16,
        background: "rgba(0,0,0,0.75)",
        borderRadius: 10,
        padding: "12px 16px",
        zIndex: 10,
        color: "#fff",
        fontFamily: "monospace",
        fontSize: 12,
        backdropFilter: "blur(8px)",
        border: "1px solid rgba(255,255,255,0.1)",
      }}
    >
      <div style={{ fontWeight: "bold", marginBottom: 8, fontSize: 13, letterSpacing: 1 }}>
        Molecular Trail Density
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <div
          style={{
            width: 120,
            height: 8,
            borderRadius: 4,
            background:
              "linear-gradient(to right, rgba(42,161,152,0.3), rgba(42,161,152,0.6), rgba(212,175,55,0.7), rgba(255,107,107,0.9))",
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", width: 120, fontSize: 10, opacity: 0.7 }}>
        <span>Low</span>
        <span>High</span>
      </div>
      <div style={{ marginTop: 10, borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }}>
        <div style={{ fontWeight: "bold", marginBottom: 6, fontSize: 13, letterSpacing: 1 }}>
          Traffic Congestion
        </div>
        {[
          { color: "#2AA198", label: "Low" },
          { color: "#D4AF37", label: "Moderate" },
          { color: "#ff6b6b", label: "Heavy" },
          { color: "#DC322F", label: "Severe" },
        ].map((item) => (
          <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
            <div style={{ width: 16, height: 3, background: item.color, borderRadius: 2 }} />
            <span style={{ fontSize: 10, opacity: 0.7 }}>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FallbackView() {
  return (
    <div className="flex items-center justify-center h-full bg-dark text-light/50">
      <div className="text-center p-8">
        <div className="text-5xl mb-4">&#x1F5FA;</div>
        <div className="font-bold text-gold mb-2">Mapbox Token Required</div>
        <div className="text-sm text-light/40">
          Add NEXT_PUBLIC_MAPBOX_TOKEN to .env.local
        </div>
        <div className="text-sm text-primaryDark mt-3">
          Munich -- Hochbunker 3D Visualization
        </div>
      </div>
    </div>
  );
}

export default function MunichMapScene() {
  const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

  if (!token) return <FallbackView />;

  return (
    <div style={{ height: "100vh", width: "100vw", position: "relative" }}>
      <Map
        antialias
        initialViewState={{
          ...MUNICH_CENTER,
          zoom: 16,
          pitch: 60,
          bearing: -20,
        }}
        mapStyle="mapbox://styles/mapbox/dark-v11"
        mapboxAccessToken={token}
      >
        {/* Traffic flow from Mapbox traffic source */}
        <TrafficLayer />

        {/* Exhaust trail heatmap along Munich roads */}
        <ExhaustHeatmap />

        {/* 3D car model */}
        <Canvas
          latitude={MUNICH_CENTER.latitude}
          longitude={MUNICH_CENTER.longitude}
        >
          <Suspense fallback={null}>
            <CarOnMap />
          </Suspense>
          <hemisphereLight
            args={["#ffffff", "#60666C"]}
            position={[1, 4.5, 3]}
            intensity={Math.PI}
          />
          <Environment preset="city" />
        </Canvas>

        {/* 3D Buildings */}
        <Buildings3D />
      </Map>

      {/* Legend overlay */}
      <MapOverlayLegend />
    </div>
  );
}

useGLTF.preload("/model/mazda_rx-7_car.glb");

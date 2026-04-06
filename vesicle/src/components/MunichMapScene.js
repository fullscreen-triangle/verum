import { useRef, useId, memo, Suspense } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF, Environment } from "@react-three/drei";
import Map, { Layer, Source } from "react-map-gl/mapbox";
import { Canvas } from "react-three-map";
import "mapbox-gl/dist/mapbox-gl.css";

// Munich Hochbunker area
const MUNICH_CENTER = { latitude: 48.1485, longitude: 11.5675 };

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
    <div style={{ height: "100vh", width: "100vw" }}>
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
        <Buildings3D />
      </Map>
    </div>
  );
}

useGLTF.preload("/model/mazda_rx-7_car.glb");

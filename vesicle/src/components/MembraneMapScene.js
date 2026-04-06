import { useRef, useId, memo, Suspense, useMemo, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF, Environment } from "@react-three/drei";
import Map, { Layer, Source } from "react-map-gl/mapbox";
import { Canvas } from "react-three-map";
import * as THREE from "three";
import "mapbox-gl/dist/mapbox-gl.css";

// Munich Hochbunker area
const MUNICH_CENTER = { latitude: 48.1485, longitude: 11.5675 };

// Membrane shader — teal/gold pulsing waves
const membraneVertex = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const membraneFragment = `
  uniform float uTime;
  uniform float uAlpha;
  uniform vec3 uColorA;
  uniform vec3 uColorB;
  varying vec2 vUv;

  void main() {
    float wave = sin(vUv.y * 40.0 + uTime * 2.0) * 0.5 + 0.5;
    float pulse = sin(uTime * 3.0) * 0.15 + 0.85;
    vec3 color = mix(uColorA, uColorB, wave) * pulse;
    gl_FragColor = vec4(color, uAlpha);
  }
`;

function MembraneCarOnMap() {
  const { scene } = useGLTF("/model/mazda_rx-7_car.glb");
  const groupRef = useRef();
  const shaderRef = useRef();

  // Create the membrane shader material
  const membraneMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 0.85 },
        uColorA: { value: new THREE.Color("#2AA198") },
        uColorB: { value: new THREE.Color("#D4AF37") },
      },
      vertexShader: membraneVertex,
      fragmentShader: membraneFragment,
      transparent: true,
      side: THREE.DoubleSide,
    });
  }, []);

  // Apply membrane shader to the largest mesh (car body)
  useEffect(() => {
    if (!scene) return;
    let largestMesh = null;
    let maxVertexCount = 0;

    scene.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const count = child.geometry.attributes.position?.count || 0;
        if (count > maxVertexCount) {
          maxVertexCount = count;
          largestMesh = child;
        }
      }
    });

    if (largestMesh) {
      largestMesh.material = membraneMaterial;
    }

    shaderRef.current = membraneMaterial;
  }, [scene, membraneMaterial]);

  // Animate shader time and rotate car
  useFrame((state, delta) => {
    if (shaderRef.current) {
      shaderRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.25;
    }
  });

  return (
    <group ref={groupRef}>
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
        <div className="text-5xl mb-4">&#x1F9EC;</div>
        <div className="font-bold text-gold mb-2">Mapbox Token Required</div>
        <div className="text-sm text-light/40">
          Add NEXT_PUBLIC_MAPBOX_TOKEN to .env.local
        </div>
        <div className="text-sm text-primaryDark mt-3">
          Munich -- Membrane Shader on Map
        </div>
      </div>
    </div>
  );
}

export default function MembraneMapScene() {
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
            <MembraneCarOnMap />
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

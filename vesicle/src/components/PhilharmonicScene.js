import { useGLTF, useScroll, ScrollControls, Scroll, Environment } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import gsap from "gsap";
import { useLayoutEffect, useRef, useMemo, Suspense } from "react";

function VehicleCircuit(props) {
  const { scene } = useGLTF("/model/v8_engine.glb");
  const ref = useRef();
  const wireRef = useRef();
  const tl = useRef();
  const scroll = useScroll();

  // Create a wireframe clone for the circuit-graph overlay
  const wireframeScene = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse((child) => {
      if (child.isMesh) {
        child.material = new THREE.MeshBasicMaterial({
          color: new THREE.Color("#2AA198"),
          wireframe: true,
          transparent: true,
          opacity: 0.12,
        });
      }
    });
    return clone;
  }, [scene]);

  useFrame(() => {
    if (tl.current) {
      tl.current.seek(scroll.offset * tl.current.duration());
    }
    // Keep wireframe in sync with main model
    if (wireRef.current && ref.current) {
      wireRef.current.position.copy(ref.current.position);
      wireRef.current.rotation.copy(ref.current.rotation);
      wireRef.current.scale.copy(ref.current.scale);
    }
  });

  useLayoutEffect(() => {
    tl.current = gsap.timeline();

    // Section 1: scale in + gentle rotation — "The Vehicle as a Circuit"
    tl.current.from(ref.current.scale, { duration: 1, x: 0, y: 0, z: 0 }, 0);
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 0.25 }, 0);

    // Section 2: rotate to reveal node structure — "Nodes and Edges"
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 0.7 }, 1);
    tl.current.to(ref.current.position, { duration: 1, x: 0.4 }, 1);

    // Section 3: tilt down, circuit perspective — "Kirchhoff's Laws"
    tl.current.to(ref.current.rotation, { duration: 1, x: -Math.PI * 0.25, y: Math.PI * 1.1 }, 2);
    tl.current.to(ref.current.position, { duration: 1, x: -0.3 }, 2);

    // Section 4: zoom out slightly — "See Inside Without Opening"
    tl.current.to(ref.current.scale, { duration: 1, x: 0.85, y: 0.85, z: 0.85 }, 3);
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 1.5 }, 3);
    tl.current.to(ref.current.position, { duration: 1, x: 0.3 }, 3);

    // Section 5: slow orbit — "Guaranteed Convergence"
    tl.current.to(ref.current.rotation, { duration: 1, x: 0, y: Math.PI * 1.8 }, 4);
    tl.current.to(ref.current.scale, { duration: 1, x: 0.9, y: 0.9, z: 0.9 }, 4);
    tl.current.to(ref.current.position, { duration: 1, x: -0.2 }, 4);

    // Section 6: final full rotation — "10/10 Validated"
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 2 }, 5);
    tl.current.to(ref.current.position, { duration: 1, x: 0 }, 5);
    tl.current.to(ref.current.scale, { duration: 1, x: 1, y: 1, z: 1 }, 5);

    // Section 7: dramatic finish — "Validated on Formula One"
    tl.current.to(ref.current.rotation, { duration: 1, x: -Math.PI * 0.15, y: Math.PI * 2.5 }, 6);
    tl.current.to(ref.current.position, { duration: 1, x: 0.2, y: -0.1 }, 6);
    tl.current.to(ref.current.scale, { duration: 1, x: 1.1, y: 1.1, z: 1.1 }, 6);
  }, []);

  return (
    <>
      <group ref={ref} {...props}>
        <primitive object={scene} scale={1} />
      </group>
      <group ref={wireRef}>
        <primitive object={wireframeScene} scale={1} />
      </group>
    </>
  );
}

function OverlayContent() {
  const sections = [
    {
      label: "Circuit Graph 1",
      title: "The Vehicle as a Circuit",
      desc: "Every car is a network of oscillators. Engine at 50 Hz. Wheels at 10 Hz. Suspension at 5 Hz. Alternator at 150 Hz. Fifteen subsystems, each oscillating, each coupled. Philharmonic reads this network as an electrical circuit graph.",
    },
    {
      label: "Circuit Graph 2",
      title: "Nodes and Edges",
      desc: "Each component is a node with potential equal to its categorical depth \u2014 the information content of its current state. Each coupling is an edge with conductance from the universal transport formula: \u039E = N\u207B\u00B9 \u03A3 \u03C4_{p,ij} g_{ij}. This single formula gives resistivity, viscosity, diffusivity, and thermal conductivity as special cases.",
    },
    {
      label: "Circuit Graph 3",
      title: "Kirchhoff\u2019s Laws for Cars",
      desc: "Energy conservation at each component gives Kirchhoff\u2019s Current Law. Thermodynamic cycle consistency around closed loops gives Kirchhoff\u2019s Voltage Law. The vehicle satisfies the same circuit equations as an electrical network \u2014 verified to 10\u207B\u00B9\u2070 precision.",
    },
    {
      label: "Circuit Graph 4",
      title: "See Inside Without Opening",
      desc: "From vibration, acoustic, thermal, and electromagnetic signals at the vehicle surface \u2014 just 33% of nodes observed \u2014 the complete internal state of every component is reconstructed. Bearing wear, injector fouling, electrical faults \u2014 all detected and localized without disassembly.",
    },
    {
      label: "Circuit Graph 5",
      title: "Guaranteed Convergence",
      desc: "The reconstruction algorithm iterates Kirchhoff propagation, backward trajectory inference, and thermodynamic projection. The Banach fixed-point theorem guarantees convergence to a unique solution. The backward trajectory of each component is time-invariant \u2014 a permanent categorical address that changes only when the component\u2019s physical state changes.",
    },
    {
      label: "Circuit Graph 6",
      title: "10/10 Validated",
      desc: "Graph construction, both Kirchhoff laws, transport formula consistency, fuzzy convergence, time-invariance, trajectory completion, contraction mapping, fault detection, and signal propagation \u2014 all validated. Every claim a theorem. Every theorem with proof.",
    },
    {
      title: "Validated on Formula One",
      label: "F1 TELEMETRY",
      desc: "Tested on real 2023 Bahrain GP data \u2014 Verstappen, 242 telemetry samples. State reconstruction with 0.9997 suspension-aero correlation. Fault detection 13 laps ahead. Tire degradation tracked. Racing line extracted from 8 qualifying laps. 4/4 tests passed.",
    },
  ];

  return (
    <Scroll html style={{ width: "100%" }}>
      {sections.map((s, i) => (
        <section
          key={i}
          style={{
            height: "100vh",
            display: "flex",
            alignItems: "center",
            padding: "0 5%",
            position: "relative",
          }}
        >
          <div
            style={{
              maxWidth: "420px",
              marginLeft: i % 2 === 0 ? "auto" : "0",
              padding: "2rem",
              background: "rgba(10,10,10,0.85)",
              borderRadius: "12px",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div
              style={{
                fontSize: "0.7rem",
                color: "#2AA198",
                letterSpacing: "0.15em",
                textTransform: "uppercase",
                marginBottom: "0.5rem",
              }}
            >
              {s.label}
            </div>
            <h2
              style={{
                fontSize: "1.5rem",
                fontWeight: "bold",
                color: "#fafafa",
                marginBottom: "0.75rem",
              }}
            >
              {s.title}
            </h2>
            <p
              style={{
                fontSize: "0.9rem",
                color: "rgba(250,250,250,0.5)",
                lineHeight: 1.7,
              }}
            >
              {s.desc}
            </p>
          </div>
        </section>
      ))}
    </Scroll>
  );
}

export default function PhilharmonicScene() {
  return (
    <Canvas
      shadows
      camera={{ position: [3, 2, 5], fov: 40 }}
      style={{ background: "#0a0a0a" }}
    >
      <ambientLight intensity={0.35} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={1.5}
        castShadow
        color="#C6A962"
      />
      <spotLight
        position={[-3, 5, 0]}
        intensity={0.8}
        color="#2AA198"
        angle={0.5}
      />
      <ScrollControls pages={7} damping={0.25}>
        <Suspense fallback={null}>
          <VehicleCircuit position={[0, 0, 0]} />
        </Suspense>
        <OverlayContent />
      </ScrollControls>
      <Environment preset="studio" />
    </Canvas>
  );
}

useGLTF.preload("/model/v8_engine.glb");

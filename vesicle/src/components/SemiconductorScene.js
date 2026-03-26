import { useGLTF, useScroll, ScrollControls, Scroll, Environment } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import gsap from "gsap";
import { useLayoutEffect, useRef, Suspense } from "react";
import * as THREE from "three";

function ChipModel(props) {
  const { scene } = useGLTF("/model/amd_ryzen.glb");
  const ref = useRef();
  const wireRef = useRef();
  const tl = useRef();
  const scroll = useScroll();

  useFrame(() => {
    tl.current.seek(scroll.offset * tl.current.duration());
    if (wireRef.current) {
      wireRef.current.rotation.copy(ref.current.rotation);
      wireRef.current.position.copy(ref.current.position);
      wireRef.current.scale.copy(ref.current.scale);
    }
  });

  useLayoutEffect(() => {
    tl.current = gsap.timeline();

    // Section 1: Classical Semiconductor — front view, scale in
    tl.current.from(ref.current.scale, { duration: 1, x: 0, y: 0, z: 0 }, 0);
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 0.25 }, 0);

    // Section 2: P-N Junction — rotate to side view, shift right
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 0.7 }, 1);
    tl.current.to(ref.current.position, { duration: 1, x: 0.6 }, 1);

    // Section 3: BMD Transistor — top-down perspective
    tl.current.to(ref.current.rotation, { duration: 1, x: -Math.PI * 0.35, y: Math.PI * 1.1 }, 2);
    tl.current.to(ref.current.position, { duration: 1, x: -0.4 }, 2);

    // Section 4: Tri-Logic Gates — zoom in slightly, angled
    tl.current.to(ref.current.rotation, { duration: 1, x: -Math.PI * 0.15, y: Math.PI * 1.5 }, 3);
    tl.current.to(ref.current.scale, { duration: 1, x: 1.15, y: 1.15, z: 1.15 }, 3);
    tl.current.to(ref.current.position, { duration: 1, x: 0.5 }, 3);

    // Section 5: ALU & Processor — pull back, dramatic angle
    tl.current.to(ref.current.rotation, { duration: 1, x: -Math.PI * 0.1, y: Math.PI * 1.85 }, 4);
    tl.current.to(ref.current.scale, { duration: 1, x: 0.85, y: 0.85, z: 0.85 }, 4);
    tl.current.to(ref.current.position, { duration: 1, x: -0.3 }, 4);

    // Section 6: 12/12 Validated — full rotation, centered
    tl.current.to(ref.current.rotation, { duration: 1, x: 0, y: Math.PI * 2.25 }, 5);
    tl.current.to(ref.current.scale, { duration: 1, x: 1, y: 1, z: 1 }, 5);
    tl.current.to(ref.current.position, { duration: 1, x: 0 }, 5);
  }, []);

  return (
    <>
      <group ref={ref} {...props}>
        <primitive object={scene} scale={1} />
      </group>
      <group ref={wireRef}>
        <mesh>
          <boxGeometry args={[1.8, 0.3, 1.8]} />
          <meshBasicMaterial color="#2AA198" wireframe opacity={0.1} transparent />
        </mesh>
      </group>
    </>
  );
}

function OverlayContent() {
  const sections = [
    {
      label: "Stage 1",
      labelColor: "#2AA198",
      title: "Classical Semiconductor",
      metric: "10\u207B\u00B3\u2079\u2079",
      metricSuffix: " tunneling probability",
      desc: "Quantum fails. Classical works. The tunneling probability through a biological membrane is 10\u207B\u00B3\u2079\u2039 \u2014 quantum coherence is physically impossible at these scales. Instead, classical phase-locking achieves 87.4% fidelity. No quantum handwaving required.",
      align: "left",
    },
    {
      label: "Stage 2",
      labelColor: "#D4AF37",
      title: "P-N Junction",
      metric: "32,680\u00D7",
      metricSuffix: " rectification ratio",
      desc: "A biological diode. Oscillatory holes (P-type) and molecular carriers (N-type) form a junction with built-in potential V_bi = 0.277 V. The Shockley equation fits with zero free parameters \u2014 every constant derived from physics.",
      align: "right",
    },
    {
      label: "Stage 3",
      labelColor: "#2AA198",
      title: "BMD Transistor",
      metric: "42.1",
      metricSuffix: " on/off ratio",
      desc: "The Biological Maxwell Demon transistor switches on pattern recognition, not voltage thresholds. The gate recognises phase-locked signatures in S-entropy space. Crossbar advantage 3.08\u00D7 over conventional architectures.",
      align: "left",
    },
    {
      label: "Stage 4",
      labelColor: "#D4AF37",
      title: "Tri-Logic Gates",
      metric: "100%",
      metricSuffix: " truth table accuracy",
      desc: "AND, OR, and XOR computed simultaneously from the same S-entropy projections. Each gate operates on a different entropy dimension (S_k, S_t, S_e). Three gates for the area of one \u2014 3\u00D7 gate density with 58% component reduction.",
      align: "right",
    },
    {
      label: "Stage 5",
      labelColor: "#2AA198",
      title: "Categorical ALU & Processor",
      metric: "15.1\u00D7",
      metricSuffix: " speedup on sorting",
      desc: "127 operations validated at 100% fidelity. The categorical ALU navigates partition space instead of manipulating bits. Energy savings of 89.5% because categorical sorting costs zero thermodynamic work \u2014 the Demon sorts for free.",
      align: "left",
    },
    {
      label: "Validated",
      labelColor: "#D4AF37",
      title: "12/12 Confirmed",
      metric: "12/12",
      metricSuffix: " stages validated",
      desc: "Every stage measured. No free parameters. No quantum coherence required. From P-N junction to full processor \u2014 a complete semiconductor architecture derived from membrane physics and confirmed by experiment.",
      align: "right",
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
              marginLeft: s.align === "right" ? "auto" : "0",
              marginRight: s.align === "left" ? "auto" : "0",
              padding: "2rem",
              background: "rgba(10,10,10,0.85)",
              borderRadius: "12px",
              border: "1px solid rgba(255,255,255,0.08)",
              backdropFilter: "blur(8px)",
            }}
          >
            <div
              style={{
                fontSize: "0.65rem",
                color: s.labelColor,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                marginBottom: "0.5rem",
                fontWeight: 600,
              }}
            >
              {s.label}
            </div>
            <h2
              style={{
                fontSize: "1.5rem",
                fontWeight: "bold",
                color: "#fafafa",
                marginBottom: "0.35rem",
                lineHeight: 1.2,
              }}
            >
              {s.title}
            </h2>
            <div style={{ marginBottom: "0.75rem" }}>
              <span
                style={{
                  fontSize: "2rem",
                  fontWeight: "bold",
                  color: s.labelColor,
                  lineHeight: 1.1,
                }}
              >
                {s.metric}
              </span>
              <span
                style={{
                  fontSize: "0.85rem",
                  color: "rgba(250,250,250,0.5)",
                  marginLeft: "0.3rem",
                }}
              >
                {s.metricSuffix}
              </span>
            </div>
            <p
              style={{
                fontSize: "0.88rem",
                color: "rgba(250,250,250,0.6)",
                lineHeight: 1.65,
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

export default function SemiconductorScene() {
  return (
    <Canvas shadows camera={{ position: [3, 2, 4], fov: 38 }} style={{ background: "#0a0a0a" }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1.8} castShadow color="#C6A962" />
      <directionalLight position={[-4, 4, -3]} intensity={0.6} color="#2AA198" />
      <spotLight position={[-3, 6, 2]} intensity={0.7} color="#2AA198" angle={0.4} penumbra={0.5} />
      <ScrollControls pages={6} damping={0.25}>
        <Suspense fallback={null}>
          <ChipModel position={[0, 0, 0]} />
        </Suspense>
        <OverlayContent />
      </ScrollControls>
      <Environment preset="studio" />
    </Canvas>
  );
}

useGLTF.preload("/model/amd_ryzen.glb");

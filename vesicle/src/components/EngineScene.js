import { useGLTF, useScroll, ScrollControls, Scroll, Environment } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import gsap from "gsap";
import { useLayoutEffect, useRef, Suspense } from "react";

function Engine(props) {
  const { scene } = useGLTF("/model/engine_with_animation.glb");
  const ref = useRef();
  const tl = useRef();
  const scroll = useScroll();

  useFrame(() => {
    tl.current.seek(scroll.offset * tl.current.duration());
  });

  useLayoutEffect(() => {
    tl.current = gsap.timeline();
    // Section 1: fade in + front view
    tl.current.from(ref.current.scale, { duration: 1, x: 0, y: 0, z: 0 }, 0);
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 0.3 }, 0);
    // Section 2: side view
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 0.8 }, 1);
    tl.current.to(ref.current.position, { duration: 1, x: 0.5 }, 1);
    // Section 3: top-down
    tl.current.to(ref.current.rotation, { duration: 1, x: -Math.PI * 0.3, y: Math.PI * 1.2 }, 2);
    // Section 4: zoom out
    tl.current.to(ref.current.scale, { duration: 1, x: 0.8, y: 0.8, z: 0.8 }, 3);
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 1.6 }, 3);
    // Section 5: final slow
    tl.current.to(ref.current.rotation, { duration: 1, y: Math.PI * 2 }, 4);
    tl.current.to(ref.current.position, { duration: 1, x: 0 }, 4);
  }, []);

  return (
    <group ref={ref} {...props}>
      <primitive object={scene} scale={1} />
    </group>
  );
}

function OverlayContent() {
  const sections = [
    { title: "Engine Combustion", freq: "~50 Hz at 3000 RPM", desc: "Each combustion cycle is a partition operation. The engine oscillates at a frequency determined by RPM — by the oscillator-processor duality, this IS computation at 50 operations per second." },
    { title: "Wheel Rotation", freq: "~10 Hz at 60 km/h", desc: "Each wheel rotation counts a categorical state transition. The wheel encoder doesn't just measure — it computes. Wheel frequency couples to engine frequency through the drivetrain." },
    { title: "CPU & Crystal Oscillators", freq: "~GHz / ~MHz", desc: "The vehicle's electronic oscillators run counting loops at billions of Hz. Their timing jitter encodes atmospheric molecular information via precision-by-difference: ΔP = T_ref - t_local." },
    { title: "Atmospheric Molecules", freq: "~10¹³ Hz", desc: "O₂ and N₂ vibrate at terahertz frequencies. Phase-locked ensembles of ~10⁴ molecules form the sensing medium. The membrane couples to these through vibrational FRET." },
    { title: "The Harmonic Network", freq: "All frequencies coupled", desc: "When oscillator frequencies have rational ratios (ω_i/ω_j = p/q), they phase-lock. The vehicle's oscillators form a harmonic coincidence network — the car's nervous system." },
  ];

  return (
    <Scroll html style={{ width: "100%" }}>
      {sections.map((s, i) => (
        <section key={i} style={{ height: "100vh", display: "flex", alignItems: "center", padding: "0 5%", position: "relative" }}>
          <div style={{ maxWidth: "400px", marginLeft: i % 2 === 0 ? "auto" : "0", padding: "2rem", background: "rgba(10,10,10,0.8)", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.08)" }}>
            <div style={{ fontSize: "0.7rem", color: "#2AA198", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: "0.5rem" }}>
              Oscillation Source {i + 1}
            </div>
            <h2 style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#fafafa", marginBottom: "0.25rem" }}>{s.title}</h2>
            <div style={{ fontSize: "1.1rem", color: "#D4AF37", fontWeight: "bold", marginBottom: "0.75rem" }}>{s.freq}</div>
            <p style={{ fontSize: "0.9rem", color: "rgba(250,250,250,0.6)", lineHeight: 1.6 }}>{s.desc}</p>
          </div>
        </section>
      ))}
    </Scroll>
  );
}

export default function EngineScene() {
  return (
    <Canvas shadows camera={{ position: [3, 2, 5], fov: 40 }} style={{ background: "#0a0a0a" }}>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 8, 5]} intensity={1.5} castShadow color="#C6A962" />
      <spotLight position={[-3, 5, 0]} intensity={0.8} color="#2AA198" angle={0.5} />
      <ScrollControls pages={5} damping={0.25}>
        <Suspense fallback={null}>
          <Engine position={[0, 0, 0]} />
        </Suspense>
        <OverlayContent />
      </ScrollControls>
      <Environment preset="warehouse" />
    </Canvas>
  );
}

useGLTF.preload("/model/engine_with_animation.glb");

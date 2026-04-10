import { useRef, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  ContactShadows,
  Environment,
  Lightformer,
  Float,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

const MODEL_PATH = "/model/mclaren_w1.glb";

function McLarenW1(props) {
  const { scene, nodes, materials } = useGLTF(MODEL_PATH);
  const ref = useRef();

  // Enhance materials once for a realistic showroom look
  if (!scene.userData._materialsEnhanced) {
    scene.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.material) {
          // Boost environment reflection for that automotive showroom feel
          child.material.envMapIntensity = 3;
          child.material.roughness = Math.min(child.material.roughness ?? 0.5, 0.35);
          child.material.metalness = Math.max(child.material.metalness ?? 0.6, 0.8);
          child.material.needsUpdate = true;
        }
      }
    });
    scene.userData._materialsEnhanced = true;
  }

  return <primitive ref={ref} object={scene} {...props} />;
}

function CameraRig({ v = new THREE.Vector3() }) {
  return useFrame((state) => {
    const t = state.clock.elapsedTime;
    state.camera.position.lerp(
      v.set(Math.sin(t / 5) * 6, 2.5, 8 + Math.cos(t / 5) * 2),
      0.025
    );
    state.camera.lookAt(0, 0, 0);
  });
}

function Podium() {
  const ref = useRef();
  useFrame((_, delta) => {
    ref.current.rotation.y += delta * 0.12;
  });
  return (
    <group ref={ref} scale={[3.5, 0.3, 3.5]} position={[0, -1.35, 0]}>
      <mesh receiveShadow>
        <cylinderGeometry args={[1, 1, 1, 64]} />
        <meshStandardMaterial metalness={0.9} roughness={0.3} color="#111" />
      </mesh>
      <mesh position={[0, 0.51, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.92, 1, 64]} />
        <meshStandardMaterial
          color={new THREE.Color(0.2, 0.8, 0.7)}
          toneMapped={false}
          roughness={0.75}
        />
      </mesh>
    </group>
  );
}

function Lightformers() {
  const group = useRef();
  useFrame(
    (state, delta) =>
      (group.current.position.z += delta * 10) > 20 &&
      (group.current.position.z = -60)
  );
  return (
    <>
      {/* Ceiling */}
      <Lightformer
        intensity={0.75}
        rotation-x={Math.PI / 2}
        position={[0, 5, -9]}
        scale={[10, 10, 1]}
      />
      <group rotation={[0, 0.5, 0]}>
        <group ref={group}>
          {[2, 0, 2, 0, 2, 0, 2, 0].map((x, i) => (
            <Lightformer
              key={i}
              form="circle"
              intensity={2}
              rotation={[Math.PI / 2, 0, 0]}
              position={[x, 4, i * 4]}
              scale={[3, 1, 1]}
            />
          ))}
        </group>
      </group>
      {/* Sides */}
      <Lightformer
        intensity={4}
        rotation-y={Math.PI / 2}
        position={[-5, 1, -1]}
        scale={[20, 0.1, 1]}
      />
      <Lightformer
        rotation-y={Math.PI / 2}
        position={[-5, -1, -1]}
        scale={[20, 0.5, 1]}
      />
      <Lightformer
        rotation-y={-Math.PI / 2}
        position={[10, 1, 0]}
        scale={[20, 1, 1]}
      />
      {/* Accent (teal) */}
      <Float speed={5} floatIntensity={2} rotationIntensity={2}>
        <Lightformer
          form="ring"
          color="#2AA198"
          intensity={8}
          scale={10}
          position={[-15, 4, -18]}
          target={[0, 0, 0]}
        />
      </Float>
      {/* Background sphere */}
      <mesh scale={100}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshBasicMaterial color="#0a0a0a" side={THREE.BackSide} />
      </mesh>
    </>
  );
}

export default function HeroScene() {
  return (
    <Suspense
      fallback={
        <div className="flex h-full w-full items-center justify-center bg-dark">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-[#2AA198] border-t-transparent" />
        </div>
      }
    >
      <Canvas
        shadows
        camera={{ position: [0, 3, 9], fov: 42 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#0a0a0a"]} />

        <group position={[0, -1, 0]} rotation={[0, -Math.PI / 6, 0]}>
          <McLarenW1 scale={[0.1, 0.1, 0.1]} />
        </group>

        <Podium />

        <hemisphereLight intensity={0.5} />
        <ContactShadows
          resolution={1024}
          frames={300}
          position={[0, -1.16, 0]}
          scale={15}
          blur={0.5}
          opacity={1}
          far={20}
        />

        {/* Decorative ground rings */}
        <mesh
          scale={4}
          position={[4, -1.161, -1.5]}
          rotation={[-Math.PI / 2, 0, Math.PI / 2.5]}
        >
          <ringGeometry args={[0.9, 1, 4, 1]} />
          <meshStandardMaterial
            color={new THREE.Color(1.1, 1.1, 1.1)}
            toneMapped={false}
            roughness={0.75}
          />
        </mesh>
        <mesh
          scale={4}
          position={[-4, -1.161, -1]}
          rotation={[-Math.PI / 2, 0, Math.PI / 2.5]}
        >
          <ringGeometry args={[0.9, 1, 3, 1]} />
          <meshStandardMaterial
            color={new THREE.Color(1.1, 1.1, 1.1)}
            toneMapped={false}
            roughness={0.75}
          />
        </mesh>

        <Environment resolution={256} background blur={1}>
          <Lightformers />
        </Environment>

        <CameraRig />
      </Canvas>
    </Suspense>
  );
}

useGLTF.preload(MODEL_PATH);

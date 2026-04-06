import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  useGLTF,
  Environment,
  ContactShadows,
  Lightformer,
  Float,
} from "@react-three/drei";
import { Suspense, useRef } from "react";
import * as THREE from "three";

function AirflowModel({
  modelPath = "/model/airshaper_demo_beta__3d_streamlines.glb",
  ...props
}) {
  const { scene } = useGLTF(modelPath);

  // Enhance materials once
  if (!scene.userData._enhanced) {
    scene.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.material) {
          child.material.envMapIntensity = 2.5;
          child.material.needsUpdate = true;
        }
      }
    });
    scene.userData._enhanced = true;
  }

  return <primitive object={scene} {...props} />;
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

export default function AirflowScene({
  modelPath = "/model/airshaper_demo_beta__3d_streamlines.glb",
  height = "60vh",
}) {
  return (
    <div
      className="w-full rounded-2xl overflow-hidden border border-light/10"
      style={{ height }}
    >
      <Suspense
        fallback={
          <div className="flex h-full w-full items-center justify-center bg-dark">
            <div className="h-10 w-10 animate-spin rounded-full border-4 border-solid border-[#2AA198] border-t-transparent" />
          </div>
        }
      >
        <Canvas
          shadows
          camera={{ position: [6, 3, 8], fov: 35 }}
          gl={{ antialias: true }}
        >
          <color attach="background" args={["#0a0a0a"]} />

          <AirflowModel
            modelPath={modelPath}
            scale={1}
            position={[0, -0.5, 0]}
          />

          <hemisphereLight intensity={0.5} />
          <ContactShadows
            resolution={1024}
            frames={300}
            position={[0, -1.16, 0]}
            scale={15}
            blur={0.5}
            opacity={0.6}
            far={20}
          />

          <Environment resolution={256} background blur={1}>
            <Lightformers />
          </Environment>

          <OrbitControls
            minPolarAngle={0}
            maxPolarAngle={Math.PI / 2}
            enableZoom
            enablePan={false}
            autoRotate
            autoRotateSpeed={0.4}
          />
        </Canvas>
      </Suspense>
    </div>
  );
}

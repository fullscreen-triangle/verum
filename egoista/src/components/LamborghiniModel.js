import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  useGLTF,
  Environment,
  ContactShadows,
} from "@react-three/drei";
import { Suspense, useRef } from "react";

const MODEL_PATH = "/model/free__lamborghini.glb";

function LamborghiniModel() {
  const { scene } = useGLTF(MODEL_PATH);
  const ref = useRef();

  return (
    <primitive
      ref={ref}
      object={scene}
      scale={1}
      position={[0, -0.5, 0]}
      dispose={null}
    />
  );
}

function Lights() {
  return (
    <>
      {/* Ambient fill */}
      <ambientLight intensity={0.3} />

      {/* Key light - warm gold to match Lamborghini branding */}
      <directionalLight
        position={[5, 8, 5]}
        intensity={1.8}
        color="#C6A962"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />

      {/* Fill light - cool blue for contrast */}
      <directionalLight
        position={[-5, 3, -3]}
        intensity={0.6}
        color="#8EBBFF"
      />

      {/* Rim light from behind for dramatic edge */}
      <pointLight position={[0, 4, -6]} intensity={0.8} color="#FFFFFF" />
    </>
  );
}

function LoadingFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-[#C6A962] border-t-transparent" />
        <p className="text-sm font-medium tracking-widest uppercase text-dark/60 dark:text-light/60">
          Loading Model...
        </p>
      </div>
    </div>
  );
}

export function LamborghiniScene() {
  return (
    <div className="h-full w-full">
      <Suspense fallback={<LoadingFallback />}>
        <Canvas
          camera={{ position: [4, 2, 5], fov: 45 }}
          style={{ background: "transparent" }}
          gl={{ alpha: true, antialias: true }}
          shadows
        >
          <Lights />

          <LamborghiniModel />

          <Environment preset="city" />

          <ContactShadows
            position={[0, -0.5, 0]}
            opacity={0.4}
            scale={10}
            blur={2.5}
            far={4}
          />

          <OrbitControls
            autoRotate
            autoRotateSpeed={0.5}
            enablePan={false}
            enableZoom={true}
            minPolarAngle={Math.PI / 4}
            maxPolarAngle={Math.PI / 2.2}
          />
        </Canvas>
      </Suspense>
    </div>
  );
}

// Preload the model
useGLTF.preload(MODEL_PATH);

export default LamborghiniScene;

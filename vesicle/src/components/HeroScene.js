import { useRef, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  ContactShadows,
  Environment,
  Lightformer,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

const MODEL_PATH = "/model/mclaren_w1.glb";

function McLarenW1(props) {
  const { scene } = useGLTF(MODEL_PATH);
  const ref = useRef();

  // Boost materials once
  if (!scene.userData._boosted) {
    scene.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.material) {
          child.material.envMapIntensity = 2;
          child.material.roughness = Math.min(child.material.roughness ?? 0.5, 0.4);
          child.material.metalness = Math.max(child.material.metalness ?? 0.6, 0.7);
          child.material.wireframe = props.wireframe || false;
          child.material.needsUpdate = true;
        }
      }
    });
    scene.userData._boosted = true;
  }

  return <primitive ref={ref} object={scene} {...props} />;
}

function CameraRig() {
  const v = useRef(new THREE.Vector3());
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    state.camera.position.lerp(
      v.current.set(Math.sin(t / 8) * 5, 2, 8 + Math.cos(t / 8) * 2),
      0.02
    );
    state.camera.lookAt(0, 0, 0);
  });
  return null;
}

function Podium() {
  const ref = useRef();
  useFrame((_, delta) => {
    ref.current.rotation.y += delta * 0.15;
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

        <group position={[0, -0.5, 0]} rotation={[0, -Math.PI / 6, 0]}>
          <McLarenW1 />
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

        <Environment resolution={512}>
          {[-9, -6, -3, 0, 3, 6, 9].map((z) => (
            <Lightformer
              key={z}
              intensity={2}
              rotation-x={Math.PI / 2}
              position={[0, 4, z]}
              scale={[10, 1, 1]}
            />
          ))}
          <Lightformer
            intensity={2}
            rotation-y={Math.PI / 2}
            position={[-50, 2, 0]}
            scale={[100, 2, 1]}
          />
          <Lightformer
            intensity={2}
            rotation-y={-Math.PI / 2}
            position={[50, 2, 0]}
            scale={[100, 2, 1]}
          />
          <Lightformer
            form="ring"
            color="#2AA198"
            intensity={10}
            scale={2}
            position={[10, 5, 10]}
            onUpdate={(self) => self.lookAt(0, 0, 0)}
          />
        </Environment>

        <CameraRig />
      </Canvas>
    </Suspense>
  );
}

useGLTF.preload(MODEL_PATH);

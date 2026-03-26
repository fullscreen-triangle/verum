import * as THREE from "three";
import { useLayoutEffect, useRef, useState, Suspense } from "react";
import { Canvas, applyProps, useFrame } from "@react-three/fiber";
import {
  PerformanceMonitor,
  AccumulativeShadows,
  RandomizedLight,
  Environment,
  Lightformer,
  Float,
  useGLTF,
  ContactShadows,
} from "@react-three/drei";
import { LayerMaterial, Color, Depth } from "lamina";

const MODEL_PATH = "/model/mclaren_w1.glb";

function McLarenW1(props) {
  const { scene, nodes, materials } = useGLTF(MODEL_PATH);
  const wireframeGroup = useRef();

  useLayoutEffect(() => {
    scene.traverse((child) => {
      if (child.isMesh) {
        child.receiveShadow = true;
        child.castShadow = true;
      }
    });

    // Premium material tweaks
    Object.values(materials).forEach((mat) => {
      if (mat.isMeshStandardMaterial || mat.isMeshPhysicalMaterial) {
        applyProps(mat, {
          envMapIntensity: 3,
          roughness: Math.min(mat.roughness || 0.5, 0.5),
          metalness: Math.max(mat.metalness || 0.6, 0.6),
        });
      }
    });
  }, [scene, nodes, materials]);

  useLayoutEffect(() => {
    if (!wireframeGroup.current) return;
    // Clear previous
    while (wireframeGroup.current.children.length) {
      wireframeGroup.current.remove(wireframeGroup.current.children[0]);
    }
    // Build wireframe overlay from each mesh
    scene.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const wireMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color("#2AA198"),
          wireframe: true,
          transparent: true,
          opacity: 0.12,
          depthWrite: false,
        });
        const wireMesh = new THREE.Mesh(child.geometry.clone(), wireMat);
        child.updateWorldMatrix(true, false);
        wireMesh.applyMatrix4(child.matrixWorld);
        wireframeGroup.current.add(wireMesh);
      }
    });
  }, [scene]);

  return (
    <group {...props}>
      <primitive object={scene} />
      <group ref={wireframeGroup} />
    </group>
  );
}

function CameraRig({ v = new THREE.Vector3() }) {
  return useFrame((state) => {
    const t = state.clock.elapsedTime;
    state.camera.position.lerp(
      v.set(Math.sin(t / 5) * 2, 0.5, 12 + Math.cos(t / 5)),
      0.05
    );
    state.camera.lookAt(0, 0, 0);
  });
}

function Lightformers({ positions = [2, 0, 2, 0, 2, 0, 2, 0] }) {
  const group = useRef();
  useFrame(
    (state, delta) =>
      (group.current.position.z += delta * 10) > 20 &&
      (group.current.position.z = -60)
  );
  return (
    <>
      <Lightformer intensity={0.75} rotation-x={Math.PI / 2} position={[0, 5, -9]} scale={[10, 10, 1]} />
      <group rotation={[0, 0.5, 0]}>
        <group ref={group}>
          {positions.map((x, i) => (
            <Lightformer key={i} form="circle" intensity={2} rotation={[Math.PI / 2, 0, 0]} position={[x, 4, i * 4]} scale={[3, 1, 1]} />
          ))}
        </group>
      </group>
      <Lightformer intensity={4} rotation-y={Math.PI / 2} position={[-5, 1, -1]} scale={[20, 0.1, 1]} />
      <Lightformer rotation-y={Math.PI / 2} position={[-5, -1, -1]} scale={[20, 0.5, 1]} />
      <Lightformer rotation-y={-Math.PI / 2} position={[10, 1, 0]} scale={[20, 1, 1]} />
      <Float speed={5} floatIntensity={2} rotationIntensity={2}>
        <Lightformer form="ring" color="#2AA198" intensity={1} scale={10} position={[-15, 4, -18]} target={[0, 0, 0]} />
      </Float>
      <mesh scale={100}>
        <sphereGeometry args={[1, 64, 64]} />
        <LayerMaterial side={THREE.BackSide}>
          <Color color="#111" alpha={1} mode="normal" />
          <Depth colorA="#2AA198" colorB="black" alpha={0.4} mode="normal" near={0} far={300} origin={[100, 100, 100]} />
        </LayerMaterial>
      </mesh>
    </>
  );
}

export default function HeroScene() {
  const [degraded, degrade] = useState(false);
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
        camera={{ position: [5, 0.5, 15], fov: 30 }}
        style={{ background: "#0a0a0a" }}
        gl={{ antialias: true }}
      >
        <spotLight position={[0, 15, 0]} angle={0.3} penumbra={1} castShadow intensity={2} shadow-bias={-0.0001} />
        <ambientLight intensity={0.5} />
        <McLarenW1 scale={1.6} position={[-0.5, -0.5, 0]} rotation={[0, Math.PI / 5, 0]} />
        <AccumulativeShadows position={[0, -1.16, 0]} frames={100} alphaTest={0.9} scale={10}>
          <RandomizedLight amount={8} radius={10} ambient={0.5} position={[1, 5, -1]} />
        </AccumulativeShadows>
        <ContactShadows position={[0, -1.16, 0]} opacity={0.4} scale={12} blur={2.5} far={4} />
        <PerformanceMonitor onDecline={() => degrade(true)} />
        <Environment frames={degraded ? 1 : Infinity} resolution={256} background blur={1}>
          <Lightformers />
        </Environment>
        <CameraRig />
      </Canvas>
    </Suspense>
  );
}

useGLTF.preload(MODEL_PATH);

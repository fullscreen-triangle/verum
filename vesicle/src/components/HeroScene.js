import * as THREE from "three";
import { useLayoutEffect, useRef, Suspense } from "react";
import { Canvas, applyProps, useFrame } from "@react-three/fiber";
import { Environment, useGLTF, OrbitControls } from "@react-three/drei";

const MODEL_PATH = "/model/mclaren_w1.glb";

function McLarenW1(props) {
  const { scene, materials } = useGLTF(MODEL_PATH);
  const wireframeGroup = useRef();

  useLayoutEffect(() => {
    scene.traverse((child) => {
      if (child.isMesh) {
        child.receiveShadow = true;
        child.castShadow = true;
      }
    });
    Object.values(materials).forEach((mat) => {
      if (mat.isMeshStandardMaterial || mat.isMeshPhysicalMaterial) {
        applyProps(mat, {
          envMapIntensity: 2,
          roughness: 0.4,
          metalness: 0.7,
        });
      }
    });
  }, [scene, materials]);

  useLayoutEffect(() => {
    if (!wireframeGroup.current) return;
    while (wireframeGroup.current.children.length) {
      wireframeGroup.current.remove(wireframeGroup.current.children[0]);
    }
    scene.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const wireMesh = new THREE.Mesh(
          child.geometry.clone(),
          new THREE.MeshBasicMaterial({
            color: "#2AA198",
            wireframe: true,
            transparent: true,
            opacity: 0.1,
            depthWrite: false,
          })
        );
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
      v.set(Math.sin(t / 5) * 3, 1, 8 + Math.cos(t / 5)),
      0.03
    );
    state.camera.lookAt(0, 0, 0);
  });
}

export default function HeroScene() {
  return (
    <Suspense
      fallback={
        <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "#0a0a0a" }}>
          <div style={{ width: 48, height: 48, border: "4px solid #2AA198", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
          <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
        </div>
      }
    >
      <Canvas
        camera={{ position: [5, 1, 10], fov: 35 }}
        style={{ background: "#0a0a0a" }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 8, 5]} intensity={1.5} color="#C6A962" />
        <directionalLight position={[-5, 3, -3]} intensity={0.5} color="#2AA198" />
        <pointLight position={[0, 5, -5]} intensity={0.8} color="#ffffff" />
        <McLarenW1 scale={1.5} position={[0, -0.5, 0]} rotation={[0, Math.PI / 5, 0]} />
        <Environment preset="city" background={false} />
        <CameraRig />
      </Canvas>
    </Suspense>
  );
}

useGLTF.preload(MODEL_PATH);

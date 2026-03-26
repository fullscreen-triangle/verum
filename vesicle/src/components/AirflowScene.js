import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment, ContactShadows } from "@react-three/drei";
import { Suspense, useRef } from "react";
import * as THREE from "three";

function AirflowModel({ modelPath = "/model/airshaper_demo_beta__3d_streamlines.glb", wireframe = false, ...props }) {
  const { scene } = useGLTF(modelPath);
  const ref = useRef();

  // Apply wireframe if requested
  scene.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
      if (wireframe) {
        child.material = new THREE.MeshBasicMaterial({
          color: "#2AA198",
          wireframe: true,
          transparent: true,
          opacity: 0.4,
        });
      }
    }
  });

  return <primitive ref={ref} object={scene} {...props} />;
}

export default function AirflowScene({
  modelPath = "/model/airshaper_demo_beta__3d_streamlines.glb",
  wireframe = false,
  height = "60vh",
}) {
  return (
    <div style={{ width: "100%", height, background: "#0a0a0a", borderRadius: 12, overflow: "hidden" }}>
      <Canvas shadows camera={{ position: [6, 3, 8], fov: 35 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.3} />
        <directionalLight position={[5, 8, 5]} intensity={1.5} castShadow color="#C6A962" />
        <spotLight position={[-5, 5, -3]} intensity={0.6} color="#2AA198" />

        <Suspense fallback={null}>
          <AirflowModel modelPath={modelPath} wireframe={wireframe} scale={1} position={[0, 0, 0]} />
        </Suspense>

        <ContactShadows position={[0, -1, 0]} opacity={0.3} scale={15} blur={2} />
        <Environment preset="city" />
        <OrbitControls autoRotate autoRotateSpeed={0.3} enablePan={false} minPolarAngle={Math.PI / 4} maxPolarAngle={Math.PI / 2.2} />
      </Canvas>
    </div>
  );
}

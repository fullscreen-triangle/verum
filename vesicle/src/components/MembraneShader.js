import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment } from "@react-three/drei";
import { Suspense, useRef, useMemo, useEffect } from "react";
import * as THREE from "three";

const MODEL_PATH = "/model/2017_aston_martin_vanquish_zagato_shooting_brake.glb";

/* -------------------------------------------------------------------------- */
/*  Membrane shader – lipid oscillation + phase-locked ensemble + S-entropy   */
/* -------------------------------------------------------------------------- */

const membraneVertexShader = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;

  void main() {
    vUv = uv;
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const membraneFragmentShader = /* glsl */ `
  uniform float uTime;
  uniform vec2 uMouse;
  uniform vec2 uResolution;

  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;

  /* ---- Colour constants ---- */
  const vec3 TEAL  = vec3(0.165, 0.631, 0.596);   /* #2AA198 - low entropy  */
  const vec3 GOLD  = vec3(0.776, 0.663, 0.384);   /* #C6A962 - high entropy */
  const vec3 BLACK = vec3(0.02, 0.02, 0.03);

  /* ---- Helpers ---- */
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  /* Smooth 2D noise for organic variation */
  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);   /* smoothstep */

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }

  /* Fractal Brownian motion */
  float fbm(vec2 p) {
    float value = 0.0;
    float amp   = 0.5;
    for (int i = 0; i < 5; i++) {
      value += amp * noise(p);
      p     *= 2.0;
      amp   *= 0.5;
    }
    return value;
  }

  void main() {
    float t = uTime;

    /* ------------------------------------------------------------------ */
    /*  1.  Oscillating lipid bilayer pattern                             */
    /*      Layered sin waves at different frequencies → cellular texture */
    /* ------------------------------------------------------------------ */
    vec2 st = vUv * 8.0;

    float lipid  = 0.0;
    lipid += 0.35 * sin(st.x *  6.2831 + t * 0.7);
    lipid += 0.25 * sin(st.y *  9.4248 - t * 0.5);
    lipid += 0.20 * sin((st.x + st.y) * 4.0 + t * 1.1);
    lipid += 0.15 * sin(length(st - 4.0) * 5.0 - t * 0.9);
    lipid += 0.10 * sin(st.x * 12.0 + st.y * 8.0 + t * 1.4);
    lipid  = lipid * 0.5 + 0.5;   /* remap to [0,1] */

    /* Add organic fbm variation */
    float organicNoise = fbm(st * 0.6 + t * 0.15);
    lipid = mix(lipid, organicNoise, 0.35);

    /* ------------------------------------------------------------------ */
    /*  2.  Phase-locked ensemble — bright spots that pulse rhythmically  */
    /* ------------------------------------------------------------------ */
    float ensemble = 0.0;
    for (int i = 0; i < 6; i++) {
      float fi    = float(i);
      vec2 center = vec2(
        0.5 + 0.35 * sin(fi * 1.17 + t * 0.3),
        0.5 + 0.35 * cos(fi * 0.93 + t * 0.25)
      );
      float dist  = length(vUv - center);
      /* Phase-locked: all spots share the same base frequency */
      float phase = sin(t * 2.5 + fi * 1.0472) * 0.5 + 0.5;
      float spot  = smoothstep(0.12, 0.0, dist) * phase;
      ensemble   += spot;
    }
    ensemble = clamp(ensemble, 0.0, 1.0);

    /* ------------------------------------------------------------------ */
    /*  3.  S-entropy colour mapping                                      */
    /*      Low entropy → teal,  High entropy → gold                     */
    /* ------------------------------------------------------------------ */
    float entropy = fbm(vUv * 5.0 + t * 0.08);
    entropy = smoothstep(0.25, 0.75, entropy);

    vec3 entropyColor = mix(TEAL, GOLD, entropy);

    /* ------------------------------------------------------------------ */
    /*  4.  Mouse interaction — membrane glows brighter near cursor       */
    /* ------------------------------------------------------------------ */
    /* uMouse is in normalised [-1,1] space; map to UV-ish [0,1] */
    vec2 mouseUV    = uMouse * 0.5 + 0.5;
    float mouseDist = length(vUv - mouseUV);
    float mouseGlow = smoothstep(0.45, 0.0, mouseDist) * 0.6;

    /* ------------------------------------------------------------------ */
    /*  5.  Compose final colour                                          */
    /* ------------------------------------------------------------------ */
    vec3 base = entropyColor * (0.4 + 0.6 * lipid);
    base += ensemble * GOLD * 0.8;                 /* ensemble pulses in gold  */
    base += mouseGlow * vec3(0.9, 0.85, 0.5);      /* warm glow near mouse    */

    /* Fresnel-like rim glow for a biological membrane feel */
    float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 3.0);
    base += fresnel * TEAL * 0.5;

    /* Overall brightness modulation to keep it alive */
    float breath = sin(t * 0.8) * 0.08 + 0.92;
    base *= breath;

    gl_FragColor = vec4(base, 0.92);
  }
`;

/* -------------------------------------------------------------------------- */
/*  Membrane-coated Lamborghini                                               */
/* -------------------------------------------------------------------------- */

function MembraneModel() {
  const { scene } = useGLTF(MODEL_PATH);
  const groupRef = useRef();
  const mouse = useRef(new THREE.Vector2(0, 0));

  /* Shared uniforms — every membrane material references the same object */
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uMouse: { value: new THREE.Vector2(0, 0) },
      uResolution: { value: new THREE.Vector2(1, 1) },
    }),
    []
  );

  /* Track mouse position */
  useEffect(() => {
    const handleMove = (e) => {
      mouse.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.current.y = -(e.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, []);

  /* Animate uniforms every frame */
  useFrame((state) => {
    uniforms.uTime.value = state.clock.getElapsedTime();
    uniforms.uMouse.value.copy(mouse.current);
  });

  /* Clone the scene and replace materials on all meshes */
  const clonedScene = useMemo(() => {
    const clone = scene.clone(true);

    clone.traverse((child) => {
      if (child.isMesh) {
        child.material = new THREE.ShaderMaterial({
          vertexShader: membraneVertexShader,
          fragmentShader: membraneFragmentShader,
          uniforms,
          transparent: true,
          side: THREE.DoubleSide,
        });
      }
    });

    return clone;
  }, [scene, uniforms]);

  return (
    <primitive
      ref={groupRef}
      object={clonedScene}
      scale={1}
      position={[0, -0.5, 0]}
      dispose={null}
    />
  );
}

/* -------------------------------------------------------------------------- */
/*  Lights                                                                    */
/* -------------------------------------------------------------------------- */

function Lights() {
  return (
    <>
      <ambientLight intensity={0.25} />
      <directionalLight position={[5, 8, 5]} intensity={1.2} color="#C6A962" />
      <directionalLight position={[-4, 3, -3]} intensity={0.5} color="#2AA198" />
      <pointLight position={[0, 5, -5]} intensity={0.6} color="#FFFFFF" />
    </>
  );
}

/* -------------------------------------------------------------------------- */
/*  Loading fallback                                                          */
/* -------------------------------------------------------------------------- */

function LoadingFallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-solid border-[#2AA198] border-t-transparent" />
        <p className="text-sm font-medium tracking-widest uppercase text-dark/60 dark:text-light/60">
          Initialising Membrane...
        </p>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Exported scene                                                            */
/* -------------------------------------------------------------------------- */

export function MembraneScene() {
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

          <MembraneModel />

          <Environment preset="city" />

          <OrbitControls
            autoRotate
            autoRotateSpeed={0.4}
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

useGLTF.preload(MODEL_PATH);

export default MembraneScene;

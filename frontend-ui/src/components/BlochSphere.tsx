import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { useSimulationStore } from "../store/useSimulationStore";

/** Compute Bloch vector from a 2-element state vector [[re0,im0],[re1,im1]] */
function stateToBloch(sv: number[][]): [number, number, number] {
  if (!sv || sv.length < 2) return [0, 0, 1];

  const [re0, im0] = sv[0] ?? [1, 0];
  const [re1, im1] = sv[1] ?? [0, 0];

  const normAlpha = Math.sqrt(re0 * re0 + im0 * im0);
  const theta = 2 * Math.acos(Math.min(1, normAlpha));

  let phi = 0;
  if (normAlpha > 1e-9) {
    // phi = arg(beta) - arg(alpha) = arg(beta/alpha)
    const betaRe = re1 * re0 + im1 * im0; // Re(beta * conj(alpha))
    const betaIm = im1 * re0 - re1 * im0; // Im(beta * conj(alpha))
    phi = Math.atan2(betaIm, betaRe);
  }

  return [
    Math.sin(theta) * Math.cos(phi),
    Math.sin(theta) * Math.sin(phi),
    Math.cos(theta),
  ];
}

function BlochPoint() {
  const meshRef = useRef<THREE.Mesh>(null);
  const targetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 1));
  const currentRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 1));

  useFrame((_, delta) => {
    const { stateVector } = useSimulationStore.getState();
    const [bx, by, bz] = stateToBloch(stateVector);
    targetRef.current.set(bx, by, bz);

    // Smooth lerp toward target
    const lerpFactor = 1 - Math.exp(-8 * delta);
    currentRef.current.lerp(targetRef.current, lerpFactor);

    if (meshRef.current) {
      meshRef.current.position.copy(currentRef.current);
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0, 1]}>
      <sphereGeometry args={[0.07, 16, 12]} />
      <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
    </mesh>
  );
}

function WireframeSphere() {
  const geometry = useMemo(() => new THREE.SphereGeometry(1, 24, 16), []);
  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial color="#1e3a5f" wireframe transparent opacity={0.35} />
    </mesh>
  );
}

function EquatorRing() {
  const points = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    const N = 64;
    for (let i = 0; i <= N; i++) {
      const angle = (i / N) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle)));
    }
    return pts;
  }, []);
  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);
  return (
    <line geometry={geometry}>
      <lineBasicMaterial color="#1e40af" transparent opacity={0.4} />
    </line>
  );
}

function AxesLines() {
  const geometry = useMemo(() => {
    const pts = [
      new THREE.Vector3(-1.3, 0, 0), new THREE.Vector3(1.3, 0, 0),
      new THREE.Vector3(0, -1.3, 0), new THREE.Vector3(0, 1.3, 0),
      new THREE.Vector3(0, 0, -1.3), new THREE.Vector3(0, 0, 1.3),
    ];
    return new THREE.BufferGeometry().setFromPoints(pts);
  }, []);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#334155" transparent opacity={0.6} />
    </lineSegments>
  );
}

function BlochTrail() {
  const trailRef = useRef<THREE.Line>(null);
  const positionsRef = useRef<Float32Array>(new Float32Array(150 * 3));
  const countRef = useRef(0);

  useFrame(() => {
    const { stateVector } = useSimulationStore.getState();
    const [bx, by, bz] = stateToBloch(stateVector);
    const idx = (countRef.current % 50) * 3;
    positionsRef.current[idx] = bx;
    positionsRef.current[idx + 1] = by;
    positionsRef.current[idx + 2] = bz;
    countRef.current++;

    if (trailRef.current) {
      const geo = trailRef.current.geometry as THREE.BufferGeometry;
      const attr = geo.attributes["position"] as THREE.BufferAttribute;
      attr.array.set(positionsRef.current);
      attr.needsUpdate = true;
    }
  });

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(150 * 3), 3));
    return geo;
  }, []);

  return (
    <line ref={trailRef} geometry={geometry}>
      <lineBasicMaterial color="#f97316" transparent opacity={0.4} />
    </line>
  );
}

export default function BlochSphere() {
  return (
    <div className="h-full w-full relative">
      <span className="absolute top-2 left-3 text-slate-400 text-xs z-10 uppercase tracking-widest pointer-events-none">
        Bloch Sphere
      </span>
      <Canvas
        camera={{ position: [2.5, 1.5, 2.5], fov: 40 }}
        style={{ background: "#0a0a18" }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[3, 3, 3]} intensity={0.8} />

        <WireframeSphere />
        <EquatorRing />
        <AxesLines />
        <BlochTrail />
        <BlochPoint />

        <Text position={[1.5, 0, 0]} fontSize={0.15} color="#94a3b8">X</Text>
        <Text position={[0, 1.5, 0]} fontSize={0.15} color="#94a3b8">Y</Text>
        <Text position={[0, 0, 1.5]} fontSize={0.15} color="#94a3b8">Z</Text>
        <Text position={[-1.5, 0, 0]} fontSize={0.15} color="#475569">-X</Text>
        <Text position={[0, -1.5, 0]} fontSize={0.15} color="#475569">-Y</Text>
        <Text position={[0, 0, -1.5]} fontSize={0.15} color="#475569">-Z</Text>

        <OrbitControls enablePan={false} minDistance={2} maxDistance={8} />
      </Canvas>
    </div>
  );
}

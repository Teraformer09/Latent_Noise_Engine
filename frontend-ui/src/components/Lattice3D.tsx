import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { useSimulationStore } from "../store/useSimulationStore";

// Max supported distance — InstancedMesh is created with MAX_N instances once.
// Qubits beyond the active d*d are scaled to zero each frame.
const MAX_D = 11;
const MAX_N = MAX_D * MAX_D; // 121
const SPACING = 1.5;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Map a probability [0,1] to a color: dim blue → bright red. */
function probToColor(p: number, color: THREE.Color): void {
  color.setRGB(
    lerp(0.05, 1.0, p),
    lerp(0.05, 0.1, p),
    lerp(0.5, 0.05, p)
  );
}

/** Pulse scale: active qubits gently pulse based on their error probability. */
function probToScale(p: number, t: number): number {
  const base = 0.32;
  const pulse = 0.06 * p * Math.sin(t * 3.0);
  return base + pulse;
}

function QubitGrid() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tempMatrix = useMemo(() => new THREE.Matrix4(), []);
  const tempColor  = useMemo(() => new THREE.Color(), []);
  const tempScale  = useMemo(() => new THREE.Vector3(), []);

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const { probabilities, d } = useSimulationStore.getState();
    const N = d * d;
    const t = clock.getElapsedTime();

    // Offset so the grid is centred regardless of d
    for (let row = 0; row < MAX_D; row++) {
      for (let col = 0; col < MAX_D; col++) {
        const idx = row * MAX_D + col;

        if (row < d && col < d) {
          // Active qubit
          const qIdx = row * d + col;
          const x = (col - (d - 1) / 2) * SPACING;
          const y = ((d - 1) / 2 - row) * SPACING;
          const p = probabilities[qIdx] ?? 0;
          const s = probToScale(p, t);

          tempScale.set(s, s, s);
          tempMatrix.compose(
            new THREE.Vector3(x, y, 0),
            new THREE.Quaternion(),
            tempScale
          );
          meshRef.current.setMatrixAt(idx, tempMatrix);

          probToColor(p, tempColor);
          meshRef.current.setColorAt(idx, tempColor);
        } else {
          // Hidden qubit — scale to zero
          tempMatrix.makeScale(0, 0, 0);
          meshRef.current.setMatrixAt(idx, tempMatrix);
        }
      }
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, MAX_N]}>
      <sphereGeometry args={[1, 20, 14]} />
      <meshStandardMaterial
        vertexColors
        roughness={0.25}
        metalness={0.6}
        emissive={new THREE.Color(0.02, 0.04, 0.1)}
        emissiveIntensity={0.4}
      />
    </instancedMesh>
  );
}

function GridLines({ d }: { d: number }) {
  const geometry = useMemo(() => {
    const pts: THREE.Vector3[] = [];

    // Horizontal edges
    for (let r = 0; r < d; r++) {
      for (let c = 0; c < d - 1; c++) {
        pts.push(
          new THREE.Vector3((c - (d - 1) / 2) * SPACING, ((d - 1) / 2 - r) * SPACING, 0),
          new THREE.Vector3((c + 1 - (d - 1) / 2) * SPACING, ((d - 1) / 2 - r) * SPACING, 0)
        );
      }
    }
    // Vertical edges
    for (let c = 0; c < d; c++) {
      for (let r = 0; r < d - 1; r++) {
        pts.push(
          new THREE.Vector3((c - (d - 1) / 2) * SPACING, ((d - 1) / 2 - r) * SPACING, 0),
          new THREE.Vector3((c - (d - 1) / 2) * SPACING, ((d - 1) / 2 - r - 1) * SPACING, 0)
        );
      }
    }

    const g = new THREE.BufferGeometry();
    g.setFromPoints(pts);
    return g;
  }, [d]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#1e3a5f" transparent opacity={0.6} />
    </lineSegments>
  );
}

function SceneContent() {
  // Read d reactively so GridLines and label update
  const d = useSimulationStore((s) => s.d);

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={1.4} color="#ffffff" />
      <pointLight position={[-5, -5, 3]} intensity={0.7} color="#60a5fa" />
      <pointLight position={[0, 0, -5]} intensity={0.3} color="#7c3aed" />

      <QubitGrid />
      <GridLines d={d} />

      <Text
        position={[0, -((d - 1) / 2) * SPACING - 0.9, 0]}
        fontSize={0.22}
        color="#475569"
      >
        {`d = ${d} surface code`}
      </Text>

      <OrbitControls enablePan={false} minDistance={2} maxDistance={20} />
    </>
  );
}

export default function Lattice3D() {
  return (
    <div className="h-full w-full relative">
      <span className="absolute top-2 left-3 text-slate-400 text-xs z-10 uppercase tracking-widest pointer-events-none">
        Surface Code Lattice
      </span>
      <Canvas
        camera={{ position: [0, 0, 9], fov: 45 }}
        style={{ background: "#0f0f1a" }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      >
        <SceneContent />
      </Canvas>
    </div>
  );
}

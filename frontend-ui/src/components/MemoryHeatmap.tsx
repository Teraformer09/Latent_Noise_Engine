import { useEffect, useRef } from "react";
import { useSimulationStore } from "../store/useSimulationStore";

const BETA = 0.5;
const MIN_SIZE = 3;

// Interpolate RGB [5,10,30] (dark navy) → [0,255,220] (bright cyan) for t in [0,1]
function cellColor(t: number): [number, number, number] {
  const r = Math.round(5 + (0 - 5) * t);
  const g = Math.round(10 + (255 - 10) * t);
  const b = Math.round(30 + (220 - 30) * t);
  return [r, g, b];
}

function buildMatrix(probs: number[]): number[][] {
  const n = Math.max(MIN_SIZE, probs.length);
  // Pad to length n if needed
  const p = [...probs];
  while (p.length < n) p.push(0);

  const matrix: number[][] = [];
  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      // Spatial decay weighted by geometric mean of qubit error probabilities
      const decay = Math.exp(-Math.abs(i - j) * BETA);
      matrix[i][j] = decay * Math.sqrt(Math.max(0, p[i]) * Math.max(0, p[j]));
    }
  }
  // Normalise to [0, 1]
  let maxVal = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) maxVal = Math.max(maxVal, matrix[i][j]);
  if (maxVal > 0) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) matrix[i][j] /= maxVal;
  }
  return matrix;
}

function drawHeatmap(canvas: HTMLCanvasElement, matrix: number[][]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const n = matrix.length;
  const w = canvas.width;
  const h = canvas.height;
  const cellW = w / n;
  const cellH = h / n;

  // Background
  ctx.fillStyle = "#0a0a18";
  ctx.fillRect(0, 0, w, h);

  // Cells
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = Math.min(1, Math.max(0, matrix[i][j]));
      const [r, g, b] = cellColor(val);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j * cellW, i * cellH, cellW, cellH);
    }
  }

  // Grid lines
  ctx.strokeStyle = "#1a1a3a";
  ctx.lineWidth = 0.5;
  for (let k = 0; k <= n; k++) {
    // vertical
    ctx.beginPath();
    ctx.moveTo(k * cellW, 0);
    ctx.lineTo(k * cellW, h);
    ctx.stroke();
    // horizontal
    ctx.beginPath();
    ctx.moveTo(0, k * cellH);
    ctx.lineTo(w, k * cellH);
    ctx.stroke();
  }

  // Title overlay
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.fillText("Memory Correlation Map", 6, 14);
}

export default function MemoryHeatmap() {
  const { probabilities } = useSimulationStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Keep canvas sized to container
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const syncSize = () => {
      canvas.width = container.offsetWidth;
      canvas.height = container.offsetHeight;
      drawHeatmap(canvas, buildMatrix(probabilities));
    };

    syncSize();
    const ro = new ResizeObserver(syncSize);
    ro.observe(container);

    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Redraw whenever probabilities change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawHeatmap(canvas, buildMatrix(probabilities));
  }, [probabilities]);

  return (
    <div
      ref={containerRef}
      className="h-full w-full bg-[#0a0a18] overflow-hidden relative"
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
      />
    </div>
  );
}

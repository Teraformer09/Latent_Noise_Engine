import { useEffect, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { useSimulationStore } from "../store/useSimulationStore";

// ---------------------------------------------------------------------------
// PSD Panel
// ---------------------------------------------------------------------------
function PSDPanel() {
  const { psdFreqs, psdAmps } = useSimulationStore();

  // Filter out zero-frequency bin for log scale
  const pairs = psdFreqs
    .map((f, i) => [f, psdAmps[i] ?? 0] as [number, number])
    .filter(([f]) => f > 0);

  const freqs = pairs.map(([f]) => f);
  const amps = pairs.map(([, a]) => a);

  // ECharts log axes require paired [x, y] data points (not separate data arrays)
  const xyData = pairs.map(([f, a]) => [f, Math.max(a, 1e-12)] as [number, number]);

  const option = {
    backgroundColor: "transparent",
    animation: false,
    grid: { top: 28, bottom: 28, left: 56, right: 12, containLabel: false },
    xAxis: {
      type: "log" as const,
      name: "Freq (Hz)",
      nameTextStyle: { color: "#64748b", fontSize: 9 },
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "log" as const,
      name: "PSD",
      nameTextStyle: { color: "#64748b", fontSize: 9 },
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "line" as const,
        data: xyData,
        smooth: false,
        symbol: "none",
        lineStyle: { color: "#a78bfa", width: 1.5 },
        areaStyle: { color: "rgba(167,139,250,0.07)" },
      },
    ],
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: "#1e293b",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 10 },
    },
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 h-full">
      <p className="text-slate-500 text-xs px-2 pt-2 uppercase tracking-widest">
        Noise PSD (1/f&#945;)
      </p>
      <div className="flex-1 min-h-0">
        <ReactECharts
          option={option}
          notMerge={false}
          lazyUpdate={true}
          style={{ height: "100%", width: "100%" }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Autocorrelation Panel — computes empirical lag-0…lag-N from psd amplitudes
// ---------------------------------------------------------------------------
const MAX_LAG = 32;

function computeAutocorr(amps: number[]): number[] {
  if (amps.length < 2) return [];
  const n = amps.length;
  const mean = amps.reduce((s, v) => s + v, 0) / n;
  const demeaned = amps.map((v) => v - mean);
  const var0 = demeaned.reduce((s, v) => s + v * v, 0) / n || 1;
  const result: number[] = [];
  const lags = Math.min(MAX_LAG, Math.floor(n / 2));
  for (let k = 0; k < lags; k++) {
    let cov = 0;
    for (let i = 0; i < n - k; i++) cov += demeaned[i] * demeaned[i + k];
    result.push(cov / (n * var0));
  }
  return result;
}

function AutocorrelationPanel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { psdAmps } = useSimulationStore();

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;
    const syncSize = () => {
      canvas.width = container.offsetWidth;
      canvas.height = container.offsetHeight;
    };
    syncSize();
    const ro = new ResizeObserver(syncSize);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, w, h);

    // Zero line
    ctx.beginPath();
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 0.5;
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    const acf = computeAutocorr(psdAmps);
    if (acf.length < 2) return;

    const barW = Math.max(2, Math.floor(w / acf.length) - 1);
    acf.forEach((r, k) => {
      const x = Math.round((k / acf.length) * w);
      const barH = Math.abs(r) * midY * 0.9;
      ctx.fillStyle = r >= 0 ? "#22d3ee" : "#f87171";
      ctx.fillRect(x, r >= 0 ? midY - barH : midY, barW, barH);
    });
  }, [psdAmps]);

  return (
    <div className="flex-1 flex flex-col min-w-0 h-full">
      <p className="text-slate-500 text-xs px-2 pt-2 uppercase tracking-widest">
        Autocorrelation (PSD)
      </p>
      <div ref={containerRef} className="flex-1 min-h-0 relative">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Threshold Gauge
// ---------------------------------------------------------------------------
const THRESHOLD = 0.1;
const GAUGE_R = 70;
const CX = 90;
const CY = 90;

function ThresholdGauge() {
  const { hazard } = useSimulationStore();
  const pct = Math.min(1, Math.max(0, hazard));

  const startAngle = -225;
  const sweepTotal = 270;
  const fillSweep = sweepTotal * pct;

  function polarToXY(angleDeg: number, r: number): [number, number] {
    const rad = (angleDeg * Math.PI) / 180;
    return [CX + r * Math.cos(rad), CY + r * Math.sin(rad)];
  }

  function arcPath(start: number, sweep: number, r: number): string {
    if (Math.abs(sweep) < 0.01) return "";
    const end = start + sweep;
    const [x1, y1] = polarToXY(start, r);
    const [x2, y2] = polarToXY(end, r);
    const large = Math.abs(sweep) > 180 ? 1 : 0;
    const dir = sweep > 0 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} ${dir} ${x2} ${y2}`;
  }

  const trackPath = arcPath(startAngle, sweepTotal, GAUGE_R);
  const fillPath = arcPath(startAngle, fillSweep, GAUGE_R);
  const isOver = pct >= THRESHOLD;

  const needleAngle = startAngle + sweepTotal * pct;
  const [nx, ny] = polarToXY(needleAngle, GAUGE_R - 10);

  return (
    <div className="flex-1 flex flex-col min-w-0 h-full">
      <p className="text-slate-500 text-xs px-2 pt-2 uppercase tracking-widest">
        Error Threshold Gauge
      </p>
      <div className="flex-1 flex items-center justify-center">
        <svg width="180" height="130" viewBox="0 0 180 130">
          <path
            d={trackPath}
            fill="none"
            stroke="#1e293b"
            strokeWidth="14"
            strokeLinecap="round"
          />
          {fillPath && (
            <path
              d={fillPath}
              fill="none"
              stroke={isOver ? "#ef4444" : "#22c55e"}
              strokeWidth="14"
              strokeLinecap="round"
              style={{ transition: "stroke 0.3s" }}
            />
          )}
          {(() => {
            const thAngle = startAngle + sweepTotal * THRESHOLD;
            const [tx1, ty1] = polarToXY(thAngle, GAUGE_R - 18);
            const [tx2, ty2] = polarToXY(thAngle, GAUGE_R + 4);
            return (
              <line
                x1={tx1}
                y1={ty1}
                x2={tx2}
                y2={ty2}
                stroke="#f59e0b"
                strokeWidth="2"
              />
            );
          })()}
          <line
            x1={CX}
            y1={CY}
            x2={nx}
            y2={ny}
            stroke="#e2e8f0"
            strokeWidth="2"
            strokeLinecap="round"
            style={{
              transformOrigin: `${CX}px ${CY}px`,
              transition: "all 0.2s",
            }}
          />
          <circle cx={CX} cy={CY} r="5" fill="#334155" />
          <text
            x={CX}
            y={CY + 22}
            textAnchor="middle"
            fontSize="16"
            fontFamily="monospace"
            fill={isOver ? "#ef4444" : "#e2e8f0"}
            fontWeight="bold"
          >
            {(pct * 100).toFixed(1)}%
          </text>
          <text
            x={CX}
            y={CY + 35}
            textAnchor="middle"
            fontSize="8"
            fill="#64748b"
            fontFamily="sans-serif"
          >
            HAZARD RATE
          </text>
          <text x="18" y="118" fontSize="9" fill="#475569" fontFamily="monospace">
            0%
          </text>
          <text x="148" y="118" fontSize="9" fill="#475569" fontFamily="monospace">
            100%
          </text>
          <text
            x={CX}
            y="14"
            textAnchor="middle"
            fontSize="8"
            fill="#f59e0b"
          >
            ⚠ {(THRESHOLD * 100).toFixed(0)}% limit
          </text>
        </svg>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// BottomPanels (container)
// ---------------------------------------------------------------------------
export default function BottomPanels() {
  return (
    <div className="h-full flex flex-row">
      <div className="flex-1 border-r border-slate-800">
        <PSDPanel />
      </div>
      <div className="flex-1 border-r border-slate-800">
        <AutocorrelationPanel />
      </div>
      <div className="flex-1">
        <ThresholdGauge />
      </div>
    </div>
  );
}

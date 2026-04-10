import ReactECharts from "echarts-for-react";
import { useSimulationStore } from "../store/useSimulationStore";

const MAX_EIGENVALUES = 8;

export default function EigenvaluePanel() {
  const { eigenvalues } = useSimulationStore();

  // Take at most 8, pad if fewer
  const raw = eigenvalues.slice(0, MAX_EIGENVALUES);

  const maxAbs = raw.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9);
  const axisMax = maxAbs * 1.15;

  const labels = raw.map((_, i) => `E${i}`);

  // Color: gradient from blue (#3b82f6) for negative to red (#ef4444) for positive
  const itemColors = raw.map((v) => {
    const t = (v + maxAbs) / (2 * maxAbs); // 0 = most negative, 1 = most positive
    const r = Math.round(59 + (239 - 59) * t);
    const g = Math.round(130 + (68 - 130) * t);
    const b = Math.round(246 + (68 - 246) * t);
    return `rgb(${r},${g},${b})`;
  });

  const option = {
    backgroundColor: "#0a0a18",
    animation: false,
    title: {
      text: "Hamiltonian Eigenvalue Spectrum",
      textStyle: {
        color: "#94a3b8",
        fontSize: 10,
        fontWeight: "normal" as const,
        fontFamily: "monospace",
      },
      top: 4,
      left: 8,
    },
    grid: {
      top: 28,
      bottom: 20,
      left: 36,
      right: 12,
      containLabel: false,
    },
    xAxis: {
      type: "value" as const,
      min: -axisMax,
      max: axisMax,
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#64748b", fontSize: 9, formatter: (v: number) => v.toFixed(2) },
      splitLine: {
        lineStyle: { color: "#1a1a3a", type: "dashed" as const },
      },
    },
    yAxis: {
      type: "category" as const,
      data: labels,
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#94a3b8", fontSize: 9, fontFamily: "monospace" },
      splitLine: { show: false },
    },
    series: [
      {
        type: "bar" as const,
        data: raw.map((v, i) => ({
          value: v,
          itemStyle: { color: itemColors[i] },
          label: {
            show: true,
            position: (v >= 0 ? "right" : "left") as "right" | "left",
            formatter: () => v.toFixed(3),
            color: "#94a3b8",
            fontSize: 8,
            fontFamily: "monospace",
          },
        })),
        barMaxWidth: 16,
      },
    ],
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: "#1e293b",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 10, fontFamily: "monospace" },
      formatter: (params: { name: string; value: number }[]) => {
        if (!params.length) return "";
        return `${params[0].name}: ${params[0].value.toFixed(6)}`;
      },
    },
  };

  return (
    <div className="h-full w-full bg-[#0a0a18] overflow-hidden">
      {raw.length === 0 ? (
        <div className="h-full flex flex-col items-center justify-center gap-1">
          <p className="text-slate-600 text-xs uppercase tracking-widest">
            Hamiltonian Eigenvalue Spectrum
          </p>
          <p className="text-slate-700 text-xs">Awaiting data…</p>
        </div>
      ) : (
        <ReactECharts
          option={option}
          notMerge={false}
          lazyUpdate={true}
          style={{ height: "100%", width: "100%" }}
        />
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { useSimulationStore } from "../store/useSimulationStore";

const MAX_POINTS = 500;

interface DataPoint {
  step: number;
  hazard: number;
  alpha: number;
}

export default function FidelityPanel() {
  const { step, hazard, alpha } = useSimulationStore();
  const bufferRef = useRef<DataPoint[]>([]);
  const [, forceRender] = useState(0);

  useEffect(() => {
    bufferRef.current = [
      ...bufferRef.current.slice(-(MAX_POINTS - 1)),
      { step, hazard, alpha },
    ];
    forceRender((n) => n + 1);
  }, [step, hazard, alpha]);

  const buf = bufferRef.current;
  const steps = buf.map((d) => d.step);
  const hazards = buf.map((d) => d.hazard);
  const alphas = buf.map((d) => d.alpha);

  const option = {
    backgroundColor: "transparent",
    animation: false,
    grid: { top: 28, bottom: 28, left: 52, right: 52, containLabel: false },
    xAxis: {
      type: "category" as const,
      data: steps,
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#64748b", fontSize: 10 },
      splitLine: { show: false },
    },
    yAxis: [
      {
        type: "value" as const,
        name: "Hazard",
        nameTextStyle: { color: "#ef4444", fontSize: 10 },
        min: 0,
        max: 1,
        axisLine: { lineStyle: { color: "#334155" } },
        axisLabel: { color: "#64748b", fontSize: 10 },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      {
        type: "value" as const,
        name: "Alpha",
        nameTextStyle: { color: "#60a5fa", fontSize: 10 },
        min: 0,
        axisLine: { lineStyle: { color: "#334155" } },
        axisLabel: { color: "#64748b", fontSize: 10 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "Hazard",
        type: "line" as const,
        data: hazards,
        yAxisIndex: 0,
        smooth: true,
        symbol: "none",
        lineStyle: { color: "#ef4444", width: 1.5 },
        areaStyle: { color: "rgba(239,68,68,0.08)" },
      },
      {
        name: "Alpha",
        type: "line" as const,
        data: alphas,
        yAxisIndex: 1,
        smooth: true,
        symbol: "none",
        lineStyle: { color: "#60a5fa", width: 1.5 },
        areaStyle: { color: "rgba(96,165,250,0.06)" },
      },
    ],
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: "#1e293b",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
  };

  return (
    <div className="h-full w-full flex flex-col">
      <p className="text-slate-500 text-xs px-3 pt-2 pb-0 uppercase tracking-widest">
        Fidelity Monitor — Hazard &amp; Alpha
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

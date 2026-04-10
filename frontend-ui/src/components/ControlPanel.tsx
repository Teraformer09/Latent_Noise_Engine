import { useState, useRef } from "react";
import { useConfigStore } from "../store/useConfigStore";
import { useSimulationStore } from "../store/useSimulationStore";

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "http://localhost:8000";

async function postAction(path: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}${path}`, { method: "POST" });
    return res.ok;
  } catch (err) {
    console.error(`[ControlPanel] POST ${path} failed:`, err);
    return false;
  }
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mb-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 w-full text-left text-xs font-semibold uppercase tracking-widest text-cyan-400 mb-2 hover:text-cyan-300 transition-colors"
      >
        <span>{open ? "▾" : "▸"}</span>
        <span>{label}</span>
      </button>
      {open && <div className="pl-1 space-y-2">{children}</div>}
    </div>
  );
}

function SliderRow({ label, min, max, step, value, onChange, format }: any) {
  const display = format ? format(value) : String(value);
  return (
    <div>
      <div className="flex justify-between items-center mb-0.5">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono text-slate-200 ml-1">{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-cyan-500"
      />
    </div>
  );
}

function SelectRow({ label, value, options, onChange }: any) {
  return (
    <div>
      <div className="flex justify-between items-center mb-0.5">
        <span className="text-xs text-slate-400">{label}</span>
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-slate-800 text-slate-200 text-xs rounded px-2 py-1 border border-slate-700"
      >
        {options.map((o: any) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

export default function ControlPanel() {
  const { qsp, noise, qec, isDirty, isSyncing, lastSyncError, updateQSP, updateNoise, updateQEC, pushToEngine, setPhiFromJSON } = useConfigStore();
  const { isConnected, isRunning, step, hazard, pauliProbs, history, exportCSV, setRunning } = useSimulationStore();
  const phiTextRef = useRef<HTMLTextAreaElement>(null);

  const handleStart = async () => {
    const ok = await postAction("/start");
    if (ok) setRunning(true);
  };

  const handleStop = async () => {
    const ok = await postAction("/stop");
    if (ok) setRunning(false);
  };

  const pauliEntries = [
    { label: "Px", value: pauliProbs.px, color: "#ef4444" },
    { label: "Py", value: pauliProbs.py, color: "#f97316" },
    { label: "Pz", value: pauliProbs.pz, color: "#3b82f6" },
    { label: "Pi", value: pauliProbs.pi, color: "#22c55e" },
  ];
  const totalPauli = pauliEntries.reduce((acc, e) => acc + e.value, 0) || 1;

  const startDisabled = isRunning || isSyncing || isDirty;

  return (
    <div className="h-full flex flex-col bg-slate-900 text-slate-100 select-none overflow-hidden">
      <div className="flex-shrink-0 px-3 pt-3 pb-2 border-b border-slate-800">
        <h1 className="text-sm font-bold leading-tight bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent mb-1">
          LATENT NOISE ENGINE
        </h1>
        <div className="flex items-center gap-2 mb-1">
          <span className={`inline-block w-2 h-2 rounded-full ${isConnected ? "bg-green-400" : "bg-red-500"}`} />
          <span className="text-xs text-slate-400">{isConnected ? "Live" : "Disconnected"}</span>
        </div>
        <div className="flex gap-3 text-xs font-mono text-slate-400">
          <span>Step: <span className="text-slate-200">{step}</span></span>
          <span>Hazard: <span className="text-slate-200">{hazard.toFixed(4)}</span></span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-0 min-h-0">
        <Section label="[A] QSP Controls">
          <SliderRow label="Degree" min={1} max={128} step={1} value={qsp.degree} onChange={(v:any) => updateQSP({degree:v})} />
          <SelectRow label="Target Function" value={qsp.targetFunction} options={[{value:"sign",label:"sign"},{value:"step",label:"step"},{value:"linear",label:"linear"}]} onChange={(v:any) => updateQSP({targetFunction:v})} />
          <SliderRow label="Rescaling α" min={0.01} max={10} step={0.01} value={qsp.rescalingFactor} onChange={(v:any) => updateQSP({rescalingFactor:v})} format={(v:any)=>v.toFixed(2)} />
          <textarea ref={phiTextRef} placeholder="[0.0, 1.57, 3.14]" rows={2} className="w-full bg-slate-800 text-slate-200 text-[10px] rounded p-1 border border-slate-700 mt-1" />
          <button onClick={() => setPhiFromJSON(phiTextRef.current?.value || "")} className="w-full py-1 mt-1 bg-slate-700 text-[10px] rounded">Validate φ JSON</button>
        </Section>

        <Section label="[B] Noise Dynamics">
          <SelectRow label="Type" value={noise.noiseType} options={[{value:"ornstein_uhlenbeck",label:"O-U"},{value:"flicker",label:"Flicker"},{value:"white",label:"White"}]} onChange={(v:any) => updateNoise({noiseType:v})} />
          <SliderRow label="τ Correlation" min={0.001} max={10} step={0.001} value={noise.tauCorr} onChange={(v:any) => updateNoise({tauCorr:v})} format={(v:any)=>v.toFixed(3)} />
          <SliderRow label="ξ Spatial" min={0.1} max={20} step={0.1} value={noise.xiSpatial} onChange={(v:any) => updateNoise({xiSpatial:v})} />
          <SliderRow label="Burst Prob" min={0} max={0.5} step={0.001} value={noise.burstProb} onChange={(v:any) => updateNoise({burstProb:v})} />
        </Section>

        <Section label="[C] QEC & Controller">
          <div className="flex gap-1 mb-2">
            {[3, 5, 7, 9, 11].map((d) => (
              <button key={d} onClick={() => updateQEC({distance:d as any})} className={`flex-1 py-1 rounded text-xs border ${qec.distance === d ? "bg-cyan-600 border-cyan-400" : "bg-slate-800 border-slate-700 text-slate-500"}`}>{d}</button>
            ))}
          </div>
          <SliderRow label="Kp" min={0} max={1000} step={1} value={qec.kp} onChange={(v:any) => updateQEC({kp:v})} />
          <SliderRow label="Target Hazard" min={0} max={1} step={0.01} value={qec.targetHazard} onChange={(v:any) => updateQEC({targetHazard:v})} />
        </Section>

        <div className="pt-2">
          <div className="flex h-2 rounded overflow-hidden">
            {pauliEntries.map(e => <div key={e.label} style={{width:`${(e.value/totalPauli)*100}%`, backgroundColor:e.color}} />)}
          </div>
        </div>
      </div>

      <div className="flex-shrink-0 px-3 pb-3 pt-2 border-t border-slate-800 space-y-2">
        <div className="flex gap-2">
          <button
            onClick={handleStart} disabled={startDisabled}
            className="flex-1 py-1.5 rounded text-xs font-bold bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-600 transition-colors"
          >
            {isDirty && !isRunning ? "⚠️ SYNC REQUIRED" : "▶ START"}
          </button>
          <button onClick={handleStop} disabled={!isRunning} className="flex-1 py-1.5 rounded text-xs font-bold bg-rose-700 hover:bg-rose-600 disabled:bg-slate-800 disabled:text-slate-600 transition-colors">■ STOP</button>
        </div>
        <button
          onClick={() => void pushToEngine()} disabled={!isDirty || isSyncing}
          className={`w-full py-2 rounded text-xs font-bold uppercase transition-all ${isDirty ? "bg-amber-500 text-slate-950 shadow-[0_0_10px_rgba(245,158,11,0.5)]" : "bg-slate-700 text-slate-500"}`}
        >
          {isSyncing ? "Updating Backend..." : "⚡ COMMIT & SYNC"}
        </button>
        {lastSyncError && <p className="text-[10px] text-red-400">{lastSyncError}</p>}
        <button
          onClick={exportCSV}
          disabled={history.length === 0}
          className="w-full py-1.5 rounded text-xs font-bold bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 transition-colors"
        >
          ↓ EXPORT CSV ({history.length} rows)
        </button>
      </div>
    </div>
  );
}

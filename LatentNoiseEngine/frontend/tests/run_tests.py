#!/usr/bin/env python3
"""
Self-contained 200+ test runner. No pytest required.
Run: python frontend/tests/run_tests.py [--fast] [--verbose]
"""
from __future__ import annotations
import sys, os, time, threading, queue as _queue, traceback, tempfile, json
import numpy as np, collections, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_FRONTEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_FRONTEND)
for p in [_ROOT, _FRONTEND]:
    if p not in sys.path: sys.path.insert(0, p)

from frontend.simulator_adapter import NoiseSimulator, DEFAULT_CONFIG
from frontend.plots.time_series import TimeSeriesPlot
from frontend.plots.stats import StatsPanel
from frontend.plots.qec_view import QECView, _MAX_TS_LEN
from frontend.viewer.pyvista_view import PyVistaViewer
from frontend.experiment.manager import ExperimentManager, save_config, load_config, config_hash

# ── harness ───────────────────────────────────────────────────────────────
_PASS=_FAIL=_SKIP=0
_RESULTS=[]

def _test(name,fn,fast=False,slow=False):
    global _PASS,_FAIL,_SKIP
    if fast and slow: _SKIP+=1; _RESULTS.append((name,"SKIP","")); return
    try: fn(); _PASS+=1; _RESULTS.append((name,"PASS",""))
    except AssertionError as e: _FAIL+=1; _RESULTS.append((name,"FAIL",str(e)))
    except Exception: _FAIL+=1; _RESULTS.append((name,"ERROR",traceback.format_exc(limit=2)))

def ok(c,m=""): assert c, m or "assertion failed"
def approx(a,b,tol=1e-6): assert abs(a-b)<tol, f"{a} != {b}"

# ── fixtures ──────────────────────────────────────────────────────────────
SIM = NoiseSimulator({"d":3,"shots":50,"seed":0})
def fresh(): return NoiseSimulator({"d":3,"shots":50,"seed":42})
def st(): return SIM.step()
def sts(n=10): return [SIM.step() for _ in range(n)]
def ts(): return TimeSeriesPlot(maxlen=500)
def sp(): return StatsPanel(maxlen=500)
def qec(): return QECView()
def vwr(d=3): return PyVistaViewer(d=d)
_TMP = tempfile.mkdtemp()
def em(): return ExperimentManager(SIM, _TMP)

# ══════════════════════════════════════════════════════════════════════════
# CAT 1 — State Flow (25)
# ══════════════════════════════════════════════════════════════════════════
def cat1(f):
    s=st()
    _test("1.01 step returns dict",           lambda: ok(isinstance(s,dict)),f)
    _test("1.02 all required keys",           lambda: [ok(k in s) for k in ["lambda_field","probabilities","hazard","alpha","qec_metrics","step"]],f)
    _test("1.03 lambda_field ndim==2",        lambda: ok(np.array(s["lambda_field"]).ndim==2),f)
    _test("1.04 lambda_field shape[1]==3",    lambda: ok(np.array(s["lambda_field"]).shape[1]==3),f)
    _test("1.05 probs >= 0",                  lambda: ok(np.all(np.array(s["probabilities"])>=0)),f)
    _test("1.06 probs <= 1",                  lambda: ok(np.all(np.array(s["probabilities"])<=1)),f)
    _test("1.07 hazard in [0,1]",             lambda: ok(0<=s["hazard"]<=1, f"h={s['hazard']}"),f)
    _test("1.08 hazard finite",               lambda: ok(np.isfinite(s["hazard"])),f)
    _test("1.09 alpha finite",                lambda: ok(np.isfinite(s["alpha"])),f)
    _test("1.10 alpha positive",              lambda: ok(s["alpha"]>0),f)
    def _11():
        r=fresh(); s1,s2=r.step(),r.step(); ok(s2["step"]>s1["step"])
    _test("1.11 step increments",_11,f)
    _test("1.12 qec_metrics dict",            lambda: ok(isinstance(s["qec_metrics"],dict)),f)
    _test("1.13 qec has 3,5,7",              lambda: [ok(d in s["qec_metrics"]) for d in [3,5,7]],f)
    _test("1.14 qec values in [0,1]",        lambda: [ok(0<=v<=1) for v in s["qec_metrics"].values()],f)
    _test("1.15 no nan in state",            lambda: (ok(not np.any(np.isnan(np.array(s["lambda_field"])))),ok(np.isfinite(s["hazard"]))),f)
    def _16():
        for i,st2 in enumerate(sts(10)): ok("hazard" in st2 and np.isfinite(st2["hazard"]))
    _test("1.16 10 steps all finite",_16,f)
    def _17():
        q=_queue.Queue(maxsize=10); r=fresh()
        for _ in range(20):
            ss=r.step()
            if not q.full(): q.put(ss)
        ok(q.qsize()<=10)
    _test("1.17 queue overflow safe",_17,f)
    def _18():
        q=_queue.Queue(maxsize=5)
        for i in range(10):
            if q.full(): q.get()
            q.put(i)
        ok(q.qsize()==5 and 9 in list(q.queue))
    _test("1.18 queue discards old",_18,f)
    def _19():
        r=fresh(); ss=r.step(); d=ss.get("d",3); ok(ss["lambda_field"].shape[0]==d*d)
    _test("1.19 lf shape matches d",_19,f)
    def _20():
        r=fresh(); ss=r.step(); d=ss.get("d",3); ok(len(ss["probabilities"])==d*d)
    _test("1.20 probs shape matches d",_20,f)
    _test("1.21 use_qsp key",                lambda: ok("use_qsp" in s),f)
    def _22():
        r=fresh(); ok(all(np.isfinite(r.step()["hazard"]) for _ in range(30)))
    _test("1.22 30 steps hazard finite",_22,f)
    _test("1.23 d field valid",              lambda: ok("d" in s and s["d"] in [3,5,7]),f)
    def _24():
        r=fresh(); steps=[r.step()["step"] for _ in range(5)]
        ok(all(steps[i]<steps[i+1] for i in range(len(steps)-1)))
    _test("1.24 steps monotone",_24,f)
    def _25():
        r=fresh(); bad=r._fallback_state("err")
        ok("hazard" in bad and "probabilities" in bad and "_error" in bad)
    _test("1.25 fallback state valid",_25,f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 2 — 3D View (30)
# ══════════════════════════════════════════════════════════════════════════
def cat2(f):
    _test("2.01 viewer constructs",         lambda: ok(vwr() is not None),f)
    _test("2.02 update no crash",           lambda: vwr().update(st()),f)
    _test("2.03 lambda mode len==9",        lambda: ok(len(vwr()._extract_field(st()))==9),f)
    def _04():
        v=vwr(); v._mode_sel.value="prob"; ok(len(v._extract_field(st()))==9)
    _test("2.04 prob mode len==9",_04,f)
    def _05():
        v=vwr(); v._mode_sel.value="qsp"; ok(len(v._extract_field(st()))==9)
    _test("2.05 qsp mode len==9",_05,f)
    def _06():
        v=vwr()
        for _ in range(5):
            field=v._extract_field(st()); ok(field.min()>=-1e-9 and field.max()<=1+1e-9)
    _test("2.06 field normalised [0,1]",_06,f)
    def _07():
        v=vwr(); s=st()
        for m in ["lambda","prob","qsp"]: v._mode_sel.value=m; v.update(s)
    _test("2.07 mode switch no crash",_07,f)
    _test("2.08 100 updates",               lambda: [vwr().update(SIM.step()) for _ in range(100)] or True,f,slow=True)
    def _09():
        v=vwr(); s=st(); s["probabilities"]=np.ones(5); v.update(s)
    _test("2.09 shape mismatch handled",_09,f)
    def _10():
        v=vwr(); s=st(); s["lambda_field"]=np.zeros_like(s["lambda_field"]); v.update(s)
    _test("2.10 zero field no crash",_10,f)
    def _11():
        v=vwr(); s=st(); s["lambda_field"]=np.full_like(s["lambda_field"],1e6); v.update(s)
    _test("2.11 extreme lambda no crash",_11,f)
    def _12():
        v=vwr(); s=st(); s["lambda_field"]=-np.abs(s["lambda_field"]); v.update(s)
    _test("2.12 negative lambda no crash",_12,f)
    def _13():
        v=vwr(d=5); s=st(); s["lambda_field"]=np.zeros((25,3)); s["probabilities"]=np.zeros(25); v.update(s)
    _test("2.13 d=5 no crash",_13,f)
    def _14():
        v=vwr(d=7); s=st(); s["lambda_field"]=np.zeros((49,3)); s["probabilities"]=np.zeros(49); v.update(s)
    _test("2.14 d=7 no crash",_14,f)
    def _15():
        v=vwr()
        for _ in range(20): ok(np.all(np.isfinite(v._extract_field(SIM.step()))))
    _test("2.15 extract always finite",_15,f)
    def _16():
        v=vwr(); s=st(); s["lambda_field"][0,0]=np.nan; v.update(s)
    _test("2.16 nan handled",_16,f)
    _test("2.17 last_state stored",         lambda: ok(setattr(x:=vwr(),'_',x.update(st())) or x._last_state is not None),f)
    _test("2.18 mode_sel exists",           lambda: ok(vwr()._mode_sel is not None),f)
    def _19():
        v=vwr(); v._mode_sel.value="prob"; s=st(); s["probabilities"]=np.zeros(9)
        ok(np.all(v._extract_field(s)==0))
    _test("2.19 zero probs → zero field",_19,f)
    def _20():
        v=vwr(); v._mode_sel.value="prob"; s=st(); s["probabilities"]=np.full(9,0.2)
        f2=v._extract_field(s); ok(f2.max()-f2.min()<1e-9)
    _test("2.20 uniform probs → no contrast",_20,f)
    def _21():
        v=vwr(); s=st(); s["lambda_field"]=np.random.randn(9,3)*100
        ok(v._extract_field(s).max()<=1+1e-9)
    _test("2.21 norm max<=1",_21,f)
    def _22():
        v=vwr(); s=st(); s["lambda_field"]=np.random.randn(9,3)*100
        ok(v._extract_field(s).min()>=-1e-9)
    _test("2.22 norm min>=0",_22,f)
    _test("2.23 d=3 default",              lambda: ok(vwr().d==3 and vwr().N==9),f)
    def _24():
        v=vwr(); s=st(); s["lambda_field"]=np.full_like(s["lambda_field"],np.inf); v.update(s)
    _test("2.24 inf lambda no crash",_24,f)
    def _25():
        v=vwr(); v._mode_sel.value="qsp"; ok(np.all(np.isfinite(v._extract_field(st()))))
    _test("2.25 qsp mode finite",_25,f)
    def _26():
        v=vwr(d=3); s=st(); s["lambda_field"]=np.random.randn(5,3)
        ok(len(v._extract_field(s))==9)
    _test("2.26 resize to N cells",_26,f)
    def _27():
        v=vwr(); t0=time.time()
        for _ in range(50): v.update(SIM.step())
        ok(time.time()-t0<30)
    _test("2.27 50 updates <30s",_27,f,slow=True)
    def _28():
        v=vwr(); s=st(); v.update(s); ok(v._last_state is s)
    _test("2.28 stores reference",_28,f)
    def _29():
        v=vwr(); s=st(); s["lambda_field"]=np.zeros((9,3)); ok(len(v._extract_field(s))==9)
    _test("2.29 empty lambda ok",_29,f)
    def _30():
        v=vwr(); s=st(); s["lambda_field"]=np.zeros(9); v.update(s)
    _test("2.30 1d lambda fallback",_30,f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 3 — Controls (25)
# ══════════════════════════════════════════════════════════════════════════
def cat3(f):
    def mk(): return fresh()
    def _01(): r=mk(); r.update_params({"base_alpha":5.0}); approx(r._config["base_alpha"],5.0)
    _test("3.01 alpha update",_01,f)
    def _02(): r=mk(); r.update_params({"sigma":0.1}); approx(r._config["sigma_temporal"],0.1)
    _test("3.02 sigma update",_02,f)
    def _03(): r=mk(); r.update_params({"distance":5}); ok(r._config["d"]==5)
    _test("3.03 distance update",_03,f)
    def _04(): r=mk(); r.update_params({"use_qsp":False}); ok(r._config["use_qsp"] is False)
    _test("3.04 qsp toggle",_04,f)
    def _05(): r=mk(); r.update_params({"target_hazard":0.2}); approx(r._config["target_hazard"],0.2)
    _test("3.05 target_hazard",_05,f)
    def _06(): r=mk(); r.update_params({"base_alpha":200.0}); ok(r._config["base_alpha"]<=100)
    _test("3.06 alpha clamped high",_06,f)
    def _07(): r=mk(); r.update_params({"base_alpha":-5.0}); ok(r._config["base_alpha"]>=0.01)
    _test("3.07 alpha clamped low",_07,f)
    def _08(): r=mk(); r.update_params({"sigma":999.0}); ok(r._config["sigma_temporal"]<=1.0)
    _test("3.08 sigma clamped",_08,f)
    def _09(): r=mk(); orig=r._config["d"]; r.update_params({"distance":99}); ok(r._config["d"]==orig)
    _test("3.09 invalid dist ignored",_09,f)
    def _10(): r=mk(); r.update_params({"base_alpha":12.34}); approx(r._config["base_alpha"],12.34)
    _test("3.10 immediate update",_10,f)
    def _11(): r=mk(); r.update_params({"burst_prob":0.05}); approx(r._config["burst_prob"],0.05)
    _test("3.11 burst_prob",_11,f)
    def _12(): r=mk(); r.update_params({"burst_prob":1.0}); ok(r._config["burst_prob"]<=0.5)
    _test("3.12 burst clamped high",_12,f)
    def _13(): r=mk(); r.update_params({"burst_prob":-1.0}); ok(r._config["burst_prob"]>=0)
    _test("3.13 burst clamped low",_13,f)
    def _14(): r=mk(); cfg=r.get_config(); ok(isinstance(cfg,dict) and "d" in cfg)
    _test("3.14 get_config",_14,f)
    def _15():
        r=mk(); errors=[]
        def run():
            for i in range(50):
                try: r.update_params({"base_alpha":float(i%10+1)})
                except Exception as e: errors.append(e)
        ts2=[threading.Thread(target=run) for _ in range(4)]
        for t in ts2: t.start()
        for t in ts2: t.join()
        ok(len(errors)==0)
    _test("3.15 thread-safe",_15,f,slow=True)
    def _16(): r=mk(); r.update_params({"base_alpha":0.1}); ok(np.isfinite(r.step()["hazard"]))
    _test("3.16 alpha change → valid step",_16,f)
    def _17(): r=mk(); r.update_params({"use_qsp":False}); ok(r.step()["use_qsp"] is False)
    _test("3.17 qsp in state",_17,f)
    def _18(): r=mk(); r.update_params({"distance":5}); ok(r.step()["d"]==5)
    _test("3.18 distance in state",_18,f)
    def _19():
        r=mk(); r.update_params({"base_alpha":3.0}); r.update_params({"sigma":0.08}); r.update_params({"distance":7})
        c=r.get_config(); approx(c["base_alpha"],3.0); approx(c["sigma_temporal"],0.08); ok(c["d"]==7)
    _test("3.19 multi-param update",_19,f)
    def _20():
        from frontend.controls.config_panel import ConfigPanel
        r=mk(); cp=ConfigPanel(r); ok(cp is not None)
    _test("3.20 ConfigPanel constructs",_20,f)
    def _21():
        from frontend.controls.config_panel import ConfigPanel
        r=mk(); cp=ConfigPanel(r); ok(cp._alpha_slider is not None and cp._sigma_slider is not None)
    _test("3.21 sliders exist",_21,f)
    def _22():
        r=mk()
        try: r.update_params({"base_alpha":float("nan")})
        except: pass
        ok(np.isfinite(r._config["base_alpha"]))
    _test("3.22 nan alpha handled",_22,f)
    def _23():
        r=mk()
        try: r.update_params({"base_alpha":float("inf")})
        except: pass
        ok(np.isfinite(r._config["base_alpha"]))
    _test("3.23 inf alpha handled",_23,f)
    def _24(): r=mk(); r.update_params({"distance":3}); ok(r._config["d"]==3)
    _test("3.24 d=3 valid",_24,f)
    def _25(): r=mk(); r.update_params({"distance":7}); ok(r._config["d"]==7)
    _test("3.25 d=7 valid",_25,f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 4 — Time Series (25)
# ══════════════════════════════════════════════════════════════════════════
def cat4(f):
    _test("4.01 update no crash",       lambda: ts().update(st()),f)
    _test("4.02 hazard appended",       lambda: ok(len((x:=ts(), x.update(st()), x)[0]._hazard)==1),f)
    _test("4.03 alpha appended",        lambda: ok(len((x:=ts(), x.update(st()), x)[0]._alpha)==1),f)
    def _04():
        t=ts(); [t.update(SIM.step()) for _ in range(100)]; ok(len(t._hazard)==100)
    _test("4.04 100 updates",_04,f)
    def _05():
        t=TimeSeriesPlot(maxlen=50); [t.update(SIM.step()) for _ in range(100)]; ok(len(t._hazard)<=50)
    _test("4.05 maxlen respected",_05,f)
    def _06():
        t=ts(); t.update(st()); ok(t._make_figure() is not None)
    _test("4.06 figure no crash",_06,f)
    def _07():
        t=ts(); t.update({"hazard":5.0,"alpha":1.0,"step":1}); ok(t._hazard[-1]<=1.0)
    _test("4.07 hazard clipped",_07,f)
    def _08():
        t=ts(); prev=len(t._hazard); t.update({"hazard":float("nan"),"alpha":1.0,"step":1}); ok(len(t._hazard)==prev)
    _test("4.08 nan skipped",_08,f)
    def _09():
        t=ts(); prev=len(t._alpha); t.update({"hazard":0.5,"alpha":float("inf"),"step":1}); ok(len(t._alpha)==prev)
    _test("4.09 inf alpha skipped",_09,f)
    def _10():
        t=ts(); t.update(st()); t.reset(); ok(len(t._hazard)==0)
    _test("4.10 reset clears",_10,f)
    _test("4.11 empty figure",          lambda: ok(ts()._make_figure() is not None),f)
    def _12():
        t=ts(); [t.update(SIM.step()) for _ in range(20)]
        steps=list(t._steps); ok(all(steps[i]<=steps[i+1] for i in range(len(steps)-1)))
    _test("4.12 monotone steps",_12,f)
    def _13():
        t=ts(); [t.update(SIM.step()) for _ in range(50)]; ok(all(0<=h<=1 for h in t._hazard))
    _test("4.13 hazard in range",_13,f)
    def _14():
        t=ts(); [t.update(SIM.step()) for _ in range(5)]; ok(len(t._make_figure().axes)==2)
    _test("4.14 figure 2 axes",_14,f)
    _test("4.15 missing hazard",        lambda: ts().update({"alpha":1.0,"step":0}),f)
    _test("4.16 missing alpha",         lambda: ts().update({"hazard":0.5,"step":0}),f)
    def _17():
        t=ts()
        for i in range(50): t.update({"hazard":0.5+0.4*np.sin(i*0.3),"alpha":1.0,"step":i})
        ok(np.array(list(t._hazard)).max()-np.array(list(t._hazard)).min()>0.1)
    _test("4.17 oscillatory visible",_17,f)
    def _18():
        t=ts()
        for i in range(10): t.update({"hazard":0.1,"alpha":1.0,"step":i})
        t.update({"hazard":0.95,"alpha":1.0,"step":10})
        ok(np.array(list(t._hazard)).max()>=0.9)
    _test("4.18 spike visible",_18,f)
    def _19():
        t=ts()
        for i in range(20): t.update({"hazard":0.8,"alpha":1.0,"step":i})
        ok(np.std(list(t._hazard))<0.01)
    _test("4.19 plateau constant",_19,f)
    def _20():
        t=TimeSeriesPlot(maxlen=1000); r=NoiseSimulator({"shots":10})
        for _ in range(500): t.update(r.step())
    _test("4.20 500 updates no crash",_20,f,slow=True)
    def _21():
        t=ts(); [t.update(SIM.step()) for _ in range(200)]; ok(len(t._hazard)<=500)
    _test("4.21 rapid bounded",_21,f)
    def _22():
        t=ts(); t.update({"hazard":0.3,"alpha":1.0,"step":99}); ok(99 in list(t._steps))
    _test("4.22 explicit step stored",_22,f)
    def _23():
        t=ts(); t.update({"hazard":-0.5,"alpha":1.0,"step":0}); ok(t._hazard[-1]>=0)
    _test("4.23 negative hazard clipped",_23,f)
    def _24():
        t=ts()
        for i in range(10): t.update({"hazard":0.3,"alpha":1.5,"step":i})
        ok(len(t._make_figure().axes[0].get_lines())>=1)
    _test("4.24 figure has lines",_24,f)
    def _25():
        t=ts(); t.update({"hazard":0.2,"alpha":-1.0,"step":0}); ok(len(t._hazard)>=0)
    _test("4.25 negative alpha no crash",_25,f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 5 — Stats (20)
# ══════════════════════════════════════════════════════════════════════════
def cat5(f):
    _test("5.01 update no crash",   lambda: sp().update(st()),f)
    _test("5.02 buffered",          lambda: ok(len((x:=sp(), x.update(st()), x)[0]._hazard_buf)==1),f)
    def _03():
        s=sp(); [s.update(SIM.step()) for _ in range(10)]; ok(s._make_figure() is not None)
    _test("5.03 figure ok",_03,f)
    def _04():
        s=sp(); acf=s._compute_acf(np.random.randn(100),30); ok(len(acf)==30)
    _test("5.04 ACF len",_04,f)
    def _05():
        s=sp(); acf=s._compute_acf(np.random.randn(100),10); approx(acf[0],1.0)
    _test("5.05 ACF lag0==1",_05,f)
    def _06():
        s=sp()
        for _ in range(50): s.update({"hazard":0.3,"step":0})
        approx(float(np.mean(list(s._hazard_buf))),0.3)
    _test("5.06 mean correct",_06,f)
    def _07():
        s=sp(); prev=len(s._hazard_buf); s.update({"hazard":float("nan"),"step":0}); ok(len(s._hazard_buf)==prev)
    _test("5.07 nan skipped",_07,f)
    def _08():
        s=sp(); s.update(st()); s.reset(); ok(len(s._hazard_buf)==0)
    _test("5.08 reset",_08,f)
    def _09():
        s=sp(); [s.update(SIM.step()) for _ in range(10)]; ok(len(s._make_figure().axes)==2)
    _test("5.09 figure 2 axes",_09,f)
    def _10():
        s=sp()
        for _ in range(30): s.update({"hazard":0.5,"step":0})
        ok(np.var(list(s._hazard_buf))<1e-10)
    _test("5.10 const → var~0",_10,f)
    def _11():
        s=sp()
        for i in range(50): s.update({"hazard":float(i%2),"step":i})
        ok(np.var(list(s._hazard_buf))>0.1)
    _test("5.11 varying → var>0.1",_11,f)
    _test("5.12 text updates",      lambda: ok("Step" in (x:=sp(), x.update(st()), x)[0]._text_pane.object),f)
    def _13():
        s=StatsPanel(maxlen=200); [s.update(SIM.step()) for _ in range(500)]; ok(len(s._hazard_buf)<=200)
    _test("5.13 bounded",_13,f,slow=True)
    def _14():
        s=sp(); ok(np.all(s._compute_acf(np.full(50,0.5),10)==0))
    _test("5.14 ACF zero var → zeros",_14,f)
    def _15():
        s=sp()
        for i in range(60): s.update({"hazard":float(i)/60,"step":i})
        ok(len(s._make_figure().axes[0].patches)>0)
    _test("5.15 histogram has patches",_15,f)
    _test("5.16 N samples in text",    lambda: ok("N samples" in (x:=sp(), x.update(st()), x)[0]._text_pane.object),f)
    def _17():
        s=sp()
        for _ in range(10): s.update({"hazard":0.2,"probabilities":np.full(9,0.05),"step":0})
        ok(len(s._prob_buf)==10)
    _test("5.17 prob_buf fills",_17,f)
    _test("5.18 empty figure",      lambda: ok(sp()._make_figure() is not None),f)
    def _19():
        s=sp(); s.update({"hazard":5.0,"step":0}); ok(s._hazard_buf[-1]<=1.0)
    _test("5.19 clip on insert",_19,f)
    def _20():
        s=sp()
        for _ in range(100): s.update({"hazard":0.42,"step":0})
        approx(float(np.mean(list(s._hazard_buf))),0.42)
    _test("5.20 mean matches raw",_20,f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 6 — QEC View (20)
# ══════════════════════════════════════════════════════════════════════════
def cat6(f):
    _test("6.01 update no crash",   lambda: qec().update(st()),f)
    def _02(): q=qec(); q.update(st()); ok(any(len(v)>0 for v in q._qec_ts.values()))
    _test("6.02 ts filled",_02,f)
    def _03(): q=qec(); [q.update(SIM.step()) for _ in range(5)]; ok(q._make_figure({}) is not None)
    _test("6.03 figure ok",_03,f)
    def _04(): q=qec(); q.update(st()); [ok(d in q._qec_ts) for d in [3,5,7]]
    _test("6.04 all distances tracked",_04,f)
    def _05():
        q=qec(); q.update({"hazard":0.5,"probabilities":np.zeros(9),"step":0,"d":3,"qec_metrics":{3:5.0,5:-1.0,7:0.3}})
        for d in [3,5,7]:
            if q._qec_ts[d]: ok(0<=q._qec_ts[d][-1]<=1)
    _test("6.05 values clipped",_05,f)
    def _06(): q=qec(); q.update(st()); ok(q._last_probs is not None)
    _test("6.06 probs stored",_06,f)
    def _07(): q=qec(); q.update(st()); ok(q._metric_text.object!="")
    _test("6.07 metric text updates",_07,f)
    def _08():
        q=qec(); [q.update(SIM.step()) for _ in range(5)]
        ok(len(q._make_figure({3:0.1,5:0.05,7:0.02}).axes[0].patches)==3)
    _test("6.08 3 bars",_08,f)
    def _09():
        q=qec()
        for _ in range(30): q.update({"hazard":0.3,"probabilities":np.full(9,0.05),"step":0,"d":3,"qec_metrics":{3:0.1,5:0.05,7:0.02}})
        ok(np.mean(list(q._qec_ts[3]))>=np.mean(list(q._qec_ts[7])))
    _test("6.09 d=3 >= d=7",_09,f)
    def _10(): q=qec(); q.update(st()); q.reset(); ok(all(len(v)==0 for v in q._qec_ts.values()))
    _test("6.10 reset",_10,f)
    def _11():
        q=qec(); prev={d:len(v) for d,v in q._qec_ts.items()}
        q.update({"hazard":0.5,"probabilities":np.zeros(9),"step":0,"d":3,"qec_metrics":{3:float("nan"),5:0.05,7:0.02}})
        ok(len(q._qec_ts[3])==prev[3]); ok(len(q._qec_ts[5])==prev[5]+1)
    _test("6.11 nan skipped",_11,f)
    def _12(): q=qec(); q.update(st()); ok(q._step>=0)
    _test("6.12 step stored",_12,f)
    def _13():
        q=qec(); [q.update(SIM.step()) for _ in range(5)]
        ok(len(q._make_figure({3:0.1,5:0.05,7:0.02}).axes)>=3)
    _test("6.13 >=3 axes",_13,f)
    def _14(): q=qec(); [q.update(SIM.step()) for _ in range(100)]
    _test("6.14 100 updates stable",_14,f,slow=True)
    def _15():
        q=qec()
        for _ in range(_MAX_TS_LEN+50): q.update(SIM.step())
        for d in [3,5,7]: ok(len(q._qec_ts[d])<=_MAX_TS_LEN)
    _test("6.15 ts bounded",_15,f,slow=True)
    def _16(): qec().update({"hazard":0.2,"probabilities":np.zeros(9),"step":0,"d":3})
    _test("6.16 missing qec_metrics ok",_16,f)
    def _17():
        q=qec(); [q.update(SIM.step()) for _ in range(10)]; ok(any(len(v)>0 for v in q._qec_ts.values()))
    _test("6.17 ts grows",_17,f)
    def _18():
        q=qec(); q.update(st()); d=q._last_d; p=np.resize(q._last_probs,d*d); ok(len(p)==d*d)
    _test("6.18 heatmap shape",_18,f)
    def _19():
        q=qec(); [q.update(SIM.step()) for _ in range(20)]
        for d in [3,5,7]: ok(all(np.isfinite(v) for v in q._qec_ts[d]))
    _test("6.19 no inf in ts",_19,f)
    _test("6.20 empty figure ok",   lambda: ok(qec()._make_figure({}) is not None),f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 7 — Experiment (20)
# ══════════════════════════════════════════════════════════════════════════
def cat7(f):
    td=tempfile.mkdtemp()
    def _01():
        p=os.path.join(td,"c1.json"); save_config({"d":3},p); ok(os.path.exists(p))
    _test("7.01 save creates file",_01,f)
    def _02():
        p=os.path.join(td,"c2.json"); save_config({"d":5,"shots":100},p)
        c=load_config(p); ok(c["d"]==5 and c["shots"]==100)
    _test("7.02 load correct",_02,f)
    def _03():
        p=os.path.join(td,"c3.json"); orig={"d":3,"alpha":1.5,"seed":99}
        save_config(orig,p); ok(load_config(p)==orig)
    _test("7.03 roundtrip",_03,f)
    def _04(): c={"d":3}; ok(config_hash(c)==config_hash(c))
    _test("7.04 hash deterministic",_04,f)
    def _05(): ok(config_hash({"d":3})!=config_hash({"d":5}))
    _test("7.05 hash differs",_05,f)
    def _06(): e=em(); ok(isinstance(e.run_batch(n_runs=2,steps_per_run=10),dict))
    _test("7.06 batch returns dict",_06,f)
    def _07(): e=em(); r=e.run_batch(n_runs=2,steps_per_run=10); ok(0<=r["mean_hazard"]<=1)
    _test("7.07 mean_hazard in [0,1]",_07,f)
    def _08(): e=em(); r=e.run_batch(n_runs=2,steps_per_run=10); ok(r["std_hazard"]>=0)
    _test("7.08 std non-negative",_08,f)
    def _09(): e=em(); r=e.run_batch(n_runs=2,steps_per_run=10); [ok(d in r["qec_mean"]) for d in [3,5,7]]
    _test("7.09 qec_mean has distances",_09,f)
    def _10(): e=em(); r=e.run_batch(n_runs=3,steps_per_run=10); ok(r["n_runs"]==3 and len(r["runs"])==3)
    _test("7.10 n_runs matches",_10,f)
    def _11():
        e=ExperimentManager(SIM,td); r=e.run_batch(n_runs=2,steps_per_run=5)
        p=e.export_csv(r,os.path.join(td,"out.csv")); ok(os.path.exists(p) and os.path.getsize(p)>0)
    _test("7.11 CSV nonempty",_11,f)
    def _12():
        e=ExperimentManager(SIM,td); r=e.run_batch(n_runs=1,steps_per_run=5)
        p=e.export_json(r,os.path.join(td,"out.json")); ok(os.path.exists(p))
    _test("7.12 JSON exists",_12,f)
    def _13():
        e=ExperimentManager(SIM,td); r=e.run_batch(n_runs=1,steps_per_run=5)
        p=e.export_json(r,os.path.join(td,"v.json"))
        with open(p) as fh: data=json.load(fh)
        ok("mean_hazard" in data)
    _test("7.13 JSON valid",_13,f)
    def _14():
        e=ExperimentManager(SIM,td); r=e.run_batch(n_runs=1,steps_per_run=5)
        p=e.export_csv(r,os.path.join(td,"h.csv"))
        with open(p) as fh: ok("hazard" in fh.readline())
    _test("7.14 CSV header",_14,f)
    def _15(): e=em(); r=e.run_batch(n_runs=2,steps_per_run=5,seeds=[10,20]); ok(10 in r["seeds"] and 20 in r["seeds"])
    _test("7.15 explicit seeds",_15,f)
    def _16(): e=em(); r=e.run_batch(n_runs=1,steps_per_run=5); ok("timestamp" in r)
    _test("7.16 timestamp",_16,f)
    def _17():
        e=em(); e.run_batch(n_runs=1,steps_per_run=5)
        lst=e.list_results(); ok(len(lst)>=1 and "mean_hazard" in lst[0])
    _test("7.17 list_results",_17,f)
    def _18(): e=em(); r=e.run_batch(n_runs=1,steps_per_run=5); ok("experiment_id" in r)
    _test("7.18 experiment_id",_18,f)
    def _19(): e=em(); r=e.run_batch(n_runs=5,steps_per_run=20); ok(r["n_runs"]==5)
    _test("7.19 large batch",_19,f,slow=True)
    def _20():
        r2=fresh(); r2.update_params({"base_alpha":7.77})
        e2=ExperimentManager(r2,td); r3=e2.run_batch(n_runs=1,steps_per_run=5)
        approx(r3["config"]["base_alpha"],7.77)
    _test("7.20 config captures alpha",_20,f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 8 — Performance (20)
# ══════════════════════════════════════════════════════════════════════════
def cat8(f):
    def _01():
        r=fresh(); t0=time.time()
        for _ in range(100): r.step()
        ok(time.time()-t0<60)
    _test("8.01 100 steps <60s",_01,f,slow=True)
    def _02():
        r=fresh(); q=_queue.Queue(maxsize=10)
        for _ in range(50):
            s=r.step()
            if not q.full(): q.put(s)
        ok(q.qsize()<=10)
    _test("8.02 queue maxsize",_02,f)
    def _03():
        r=fresh(); q=_queue.Queue(maxsize=10); t2=ts(); s2=sp(); q2=qec()
        for _ in range(20):
            ss=r.step()
            if not q.full(): q.put(ss)
        while not q.empty():
            ss=q.get(); t2.update(ss); s2.update(ss); q2.update(ss)
    _test("8.03 ui update loop",_03,f)
    def _04():
        r=fresh(); errors=[]
        def sl():
            for _ in range(100):
                try: r.step()
                except Exception as e: errors.append(e)
        def pl():
            for i in range(100):
                try: r.update_params({"base_alpha":float(i%10+1)})
                except Exception as e: errors.append(e)
        t1=threading.Thread(target=sl); t2=threading.Thread(target=pl)
        t1.start(); t2.start(); t1.join(); t2.join()
        ok(len(errors)==0)
    _test("8.04 concurrent no errors",_04,f,slow=True)
    def _05():
        r=fresh()
        for _ in range(1000): ok(r.step() is not None)
    _test("8.05 1000 steps",_05,f,slow=True)
    def _06():
        r=fresh(); init=threading.active_count()
        for _ in range(50): r.step()
        ok(threading.active_count()-init<5)
    _test("8.06 no thread explosion",_06,f)
    def _07():
        t=TimeSeriesPlot(maxlen=500)
        for _ in range(1000): t.update(SIM.step())
        ok(len(t._hazard)<=500)
    _test("8.07 deque bounded",_07,f,slow=True)
    def _08():
        s=StatsPanel(maxlen=200)
        for _ in range(500): s.update(SIM.step())
        ok(len(s._hazard_buf)<=200)
    _test("8.08 stats bounded",_08,f,slow=True)
    def _09():
        q=qec()
        for _ in range(_MAX_TS_LEN+50): q.update(SIM.step())
        for d in [3,5,7]: ok(len(q._qec_ts[d])<=_MAX_TS_LEN)
    _test("8.09 qec ts bounded",_09,f,slow=True)
    def _10():
        r=fresh()
        for _ in range(10): ok("hazard" in r._fallback_state("x"))
    _test("8.10 fallback always valid",_10,f)
    def _11():
        r=fresh()
        for _ in range(100):
            s=r.step(); ok(np.isfinite(s["hazard"]) and np.isfinite(s["alpha"]))
    _test("8.11 no nan 100 steps",_11,f)
    def _12():
        r=fresh()
        for _ in range(50):
            s=r.step(); p=s["probabilities"]
            ok(np.all(np.isfinite(p)) and np.all(p>=0) and np.all(p<=1))
    _test("8.12 probs bounded",_12,f)
    def _13():
        r=fresh(); time.sleep(0.05); ok(isinstance(r.get_config(),dict))
    _test("8.13 idle stable",_13,f)
    def _14():
        r=fresh(); r._fallback_state("x"); ok("hazard" in r.step())
    _test("8.14 recovery after fallback",_14,f)
    def _15():
        r=fresh(); q=_queue.Queue(maxsize=5); stop=threading.Event()
        def loop():
            while not stop.is_set():
                s=r.step()
                if not q.full(): q.put(s)
                time.sleep(0.02)
        t=threading.Thread(target=loop,daemon=True)
        t.start(); time.sleep(0.15); stop.set(); t.join(timeout=2)
        ok(q.qsize()>0)
    _test("8.15 bg thread feeds queue",_15,f)
    def _16():
        td2=tempfile.mkdtemp(); e=ExperimentManager(SIM,td2); r=e.run_batch(n_runs=2,steps_per_run=20)
        t0=time.time()
        e.export_csv(r,os.path.join(td2,"s.csv")); e.export_json(r,os.path.join(td2,"s.json"))
        ok(time.time()-t0<10)
    _test("8.16 export fast",_16,f)
    def _17():
        t=ts(); t0=time.time()
        for _ in range(100): t.update(SIM.step())
        ok(time.time()-t0<30)
    _test("8.17 100 plot updates <30s",_17,f,slow=True)
    def _18():
        r=fresh()
        for _ in range(200):
            s=r.step(); ok(not np.any(np.isinf(s["probabilities"])))
    _test("8.18 no overflow 200 steps",_18,f,slow=True)
    def _19():
        sims=[NoiseSimulator({"shots":20,"seed":i}) for i in range(5)]
        for sim2 in sims: ok(np.isfinite(sim2.step()["hazard"]))
    _test("8.19 5 simultaneous sims",_19,f)
    def _20():
        r=NoiseSimulator({"d":7,"shots":30})
        for _ in range(20): ok(np.isfinite(r.step()["hazard"]))
    _test("8.20 d=7 stable",_20,f)

# ══════════════════════════════════════════════════════════════════════════
# CAT 9 — Causality (25)
# ══════════════════════════════════════════════════════════════════════════
def cat9(f):
    def _01():
        r=fresh(); r.update_params({"sigma":0.001}); sl=[r.step() for _ in range(20)]
        r.update_params({"sigma":0.4}); sh=[r.step() for _ in range(20)]
        ok(all(np.isfinite(s["probabilities"].mean()) for s in sl+sh))
    _test("9.01 sigma change → valid probs",_01,f)
    def _02():
        r=fresh(); r.update_params({"use_qsp":True}); on=r.step()
        r.update_params({"use_qsp":False}); off=r.step()
        ok(np.isfinite(on["hazard"]) and np.isfinite(off["hazard"]))
    _test("9.02 QSP on/off both valid",_02,f)
    def _03():
        r=fresh(); r.update_params({"use_qsp":True}); ok(r.step()["use_qsp"] is True)
        r.update_params({"use_qsp":False}); ok(r.step()["use_qsp"] is False)
    _test("9.03 QSP flag in state",_03,f)
    def _04():
        r=fresh(); r.update_params({"distance":3}); ok(r.step()["d"]==3)
        r.update_params({"distance":7}); ok(r.step()["d"]==7)
    _test("9.04 d in state",_04,f)
    def _05():
        r=fresh(); v3,v7=[],[]
        for _ in range(30):
            s=r.step(); qc=s.get("qec_metrics",{})
            if 3 in qc: v3.append(qc[3])
            if 7 in qc: v7.append(qc[7])
        if v3 and v7: ok(np.mean(v3)>=np.mean(v7))
    _test("9.05 pL(d=3) >= pL(d=7)",_05,f)
    def _06():
        r=fresh(); ok(all(np.isfinite(r.step()["hazard"]) for _ in range(50)))
    _test("9.06 hazard finite 50 steps",_06,f)
    def _07():
        r=fresh()
        for _ in range(20): ok(np.all(np.isfinite(r.step()["probabilities"])))
    _test("9.07 probs finite",_07,f)
    def _08():
        r=fresh(); s=r.step()
        for k in ["lambda_field","probabilities","hazard","alpha","qec_metrics","step","use_qsp","d"]:
            ok(k in s, f"missing {k}")
    _test("9.08 all pipeline keys",_08,f)
    def _09():
        r=fresh(); s=r.step(); ok(s["lambda_field"].shape[0]==len(s["probabilities"]))
    _test("9.09 lambda N == probs N",_09,f)
    def _10():
        r=fresh()
        for _ in range(100):
            s=r.step(); ok(np.isfinite(s["hazard"]) and s["hazard"]<=1)
    _test("9.10 no diverge 100 steps",_10,f,slow=True)
    def _11():
        r=fresh(); alphas=[r.step()["alpha"] for _ in range(50)]
        ok(not all(a==alphas[0] for a in alphas))
    _test("9.11 alpha varies",_11,f)
    def _12():
        v=vwr(); v._mode_sel.value="lambda"; r=fresh()
        r.update_params({"sigma":0.001}); f1=v._extract_field(r.step())
        r.update_params({"sigma":0.3});   f2=v._extract_field(r.step())
        ok(np.all(np.isfinite(f1)) and np.all(np.isfinite(f2)))
    _test("9.12 viz finite after sigma change",_12,f)
    def _13():
        q2=qec(); r=fresh(); s=r.step(); q2.update(s)
        d=s["d"]; ok(len(np.resize(q2._last_probs,d*d))==d*d)
    _test("9.13 qec heatmap shape",_13,f)
    def _14():
        s2=sp()
        for _ in range(100): s2.update({"hazard":0.42,"step":0})
        approx(float(np.mean(list(s2._hazard_buf))),0.42)
    _test("9.14 stats mean exact",_14,f)
    def _15():
        t=ts()
        for i in range(20): t.update({"hazard":0.5,"alpha":2.0,"step":i})
        ok(np.std(list(t._hazard))<1e-10)
    _test("9.15 plateau std~0",_15,f)
    def _16():
        r=fresh(); r.update_params({"sigma":0.4,"burst_prob":0.1})
        ok(any(r.step()["hazard"]>0 for _ in range(30)))
    _test("9.16 high noise → nonzero hazard",_16,f)
    def _17():
        r=fresh(); r.update_params({"sigma":0.001,"burst_prob":0.0,"base_alpha":0.1})
        ok(all(0<=r.step()["hazard"]<=1 for _ in range(30)))
    _test("9.17 low noise → hazard in range",_17,f)
    def _18():
        r=fresh(); t=ts(); s2=sp(); q2=qec()
        states=[r.step() for _ in range(20)]
        for ss in states: t.update(ss); s2.update(ss); q2.update(ss)
        ok(len(t._hazard)==len(list(s2._hazard_buf)))
    _test("9.18 panels in sync",_18,f)
    def _19():
        r=fresh(); r.update_params({"base_alpha":5.0})
        e2=ExperimentManager(r,_TMP); r2=e2.run_batch(n_runs=2,steps_per_run=10)
        approx(r2["config"]["base_alpha"],5.0)
    _test("9.19 experiment captures alpha",_19,f)
    def _20():
        q2=qec(); [q2.update(SIM.step()) for _ in range(50)]
        m3=np.mean(list(q2._qec_ts[3])) if q2._qec_ts[3] else 0
        m7=np.mean(list(q2._qec_ts[7])) if q2._qec_ts[7] else 0
        ok(m3>=m7)
    _test("9.20 QEC d=3>=d=7 mean",_20,f)
    def _21():
        r=fresh(); t=ts(); s2=sp(); q2=qec()
        for _ in range(200): ss=r.step(); t.update(ss); s2.update(ss); q2.update(ss)
        ok(all(np.isfinite(h) for h in t._hazard))
        ok(all(np.isfinite(h) for h in s2._hazard_buf))
    _test("9.21 all finite after 200",_21,f,slow=True)
    def _22():
        r=fresh(); s=r.step()
        ok(s["lambda_field"].shape[1]==3)
        ok(np.all(s["probabilities"]>=0))
        ok(all(d in s["qec_metrics"] for d in [3,5,7]))
        ok(0<=s["hazard"]<=1)
        ok(np.isfinite(s["alpha"]))
    _test("9.22 pipeline diagram",_22,f)
    def _23():
        v=vwr(); v._mode_sel.value="prob"; r=fresh()
        r.update_params({"use_qsp":True}); ok(np.all(np.isfinite(v._extract_field(r.step()))))
    _test("9.23 QSP on prob mode finite",_23,f)
    def _24():
        f1=fresh(); f1.update_params({"burst_prob":0.0}); s1=f1.step()
        f2=fresh(); f2.update_params({"burst_prob":0.15}); s2=f2.step()
        ok(np.isfinite(s1["hazard"]) and np.isfinite(s2["hazard"]))
    _test("9.24 burst vs no-burst valid",_24,f)
    def _25():
        r=fresh()
        for _ in range(10): r.update_params({"base_alpha":float(np.random.rand()*10+1)})
        ok(np.isfinite(r.get_config()["base_alpha"]))
    _test("9.25 config stable rapid changes",_25,f)

# ══════════════════════════════════════════════════════════════════════════
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fast",action="store_true")
    parser.add_argument("--verbose",action="store_true")
    args=parser.parse_args()

    print("="*70)
    print("  Latent Noise Engine — Frontend Test Suite  (200+ tests)")
    print("="*70)
    t0=time.time()

    cat1(args.fast); cat2(args.fast); cat3(args.fast)
    cat4(args.fast); cat5(args.fast); cat6(args.fast)
    cat7(args.fast); cat8(args.fast); cat9(args.fast)

    elapsed=time.time()-t0
    total=_PASS+_FAIL+_SKIP
    print()
    for name,status,detail in _RESULTS:
        if args.verbose or status in ("FAIL","ERROR"):
            icon={"PASS":"✓","FAIL":"✗","SKIP":"○","ERROR":"✗"}.get(status,"?")
            line=f"  {icon} [{status}] {name}"
            if detail and status in ("FAIL","ERROR"):
                short=detail.strip().split("\n")[-1][:80]
                line+=f"\n       → {short}"
            print(line)
    print()
    print("─"*70)
    print(f"  Total:{total}  PASS:{_PASS}  FAIL:{_FAIL}  SKIP:{_SKIP}  Time:{elapsed:.1f}s")
    print("─"*70)
    if _FAIL==0:
        print("  ✅  ALL TESTS PASSED — deployment ready")
    else:
        print(f"  ❌  {_FAIL} FAILED")
        sys.exit(1)

if __name__=="__main__":
    main()

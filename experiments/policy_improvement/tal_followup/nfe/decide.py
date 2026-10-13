"""validate: batch-96 validation verdict. project: NFE-100 cost vs the 15 GPU-h cap."""
import json, sys, numpy as np
sys.path.insert(0,"experiments/policy_improvement/tal_followup/uwm")
P2="experiments/policy_improvement/phase2/tf_%s_R%d_diagnostics.json"; SETS=["E0","E1s","E2s","E3s","E4s"]
def macro(name):
    out=[]; secs=[]
    for r in (1,2,3):
        c=json.load(open(P2%(name,r)))["control"]
        assert all(c[s]["status"]=="VALID" and c[s]["n"]==96 for s in SETS), "not all VALID/n=96"
        out.append(np.mean([c[s]["success"] for s in SETS])); secs+=[c[s]["seconds"] for s in SETS]
    return out, secs
if sys.argv[1]=="validate":
    b16,_=macro("ctl10k"); b96,secs=macro("ctl10k_b96")
    d=100*(np.mean(b96)-np.mean(b16)); ok=abs(d)<2.48
    v={"b16_macro":b16,"b96_macro":b96,"delta_pp":d,"accept":bool(ok),"b96_set_seconds_mean":float(np.mean(secs))}
    json.dump(v,open("experiments/policy_improvement/tal_followup/nfe/VALIDATION.json","w"),indent=1); print(v); sys.exit(0 if ok else 1)
else:
    from program_spend import spent
    v=json.load(open("experiments/policy_improvement/tal_followup/nfe/VALIDATION.json"))
    t=json.load(open("experiments/policy_improvement/tal_followup/nfe/timing_b96.json"))
    sim=v["b96_set_seconds_mean"]-100*t["ctl10k_nfe4"]          # per-set non-sampler time at batch 96
    per={m:(sim+100*t["%s_nfe100"%m])/3600 for m in ("ctl10k","qaction10k")}
    need=15*sum(per.values())*1.1                                # 2 models x 5 sets x R=3, +10% margin
    s=spent(); ok=s+need<=15.0
    pj={"spent":s,"per_set_gpu_h":per,"need_with_margin":need,"total":s+need,"fits":bool(ok)}
    json.dump(pj,open("experiments/policy_improvement/tal_followup/nfe/PROJECTION.json","w"),indent=1); print(pj); sys.exit(0 if ok else 1)

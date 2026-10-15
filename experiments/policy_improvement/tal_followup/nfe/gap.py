"""Gap analysis on batch-96 runs. usage: gap.py R [r1check]"""
import json, sys, numpy as np
P2="experiments/policy_improvement/phase2/tf_%s_R%d_diagnostics.json"; S=["E0","E1s","E2s","E3s","E4s"]
R=int(sys.argv[1])
def get(n):
    M=np.zeros((R,5))
    for r in range(R):
        c=json.load(open(P2%(n,r+1)))["control"]
        for j,s in enumerate(S):
            assert c[s]["status"]=="VALID" and c[s]["n"]==96, (n,r+1,s)
            M[r,j]=100*c[s]["success"]
    return M
out={}
for lab,c,q in (("NFE4","ctl10k_b96","qaction10k_b96"),("NFE100","ctl10k_b96n100","qaction10k_b96n100")):
    try: C,Q=get(c),get(q)
    except Exception as e: print(lab,"incomplete",e); continue
    g=C-Q
    out[lab]={"ctl_macro":C.mean(),"qa_macro":Q.mean(),"gap":g.mean(),"gap_per_set":g.mean(0).tolist(),
              "gap_per_R":g.mean(1).tolist(),"ctl_per_R":C.mean(1).tolist(),"qa_per_R":Q.mean(1).tolist()}
    print("%-6s control %.2f  direct %.2f  gap %+.2f pp | per set %s | per R %s"%(lab,C.mean(),Q.mean(),g.mean(),
          " ".join("%+.1f"%x for x in g.mean(0))," ".join("%+.2f"%x for x in g.mean(1))))
json.dump(out,open("experiments/policy_improvement/tal_followup/nfe/GAP_R%d.json"%R,"w"),indent=1)
if len(sys.argv)>2:   # R1 stop rule: gap exploded (>=8 pp) and consistent across all 5 sets -> stop
    g=out["NFE100"]; exploded = g["gap"]>=8 and all(x>0 for x in g["gap_per_set"])
    print("R1 rule: exploded=%s"%exploded)
    if exploded: sys.exit(10)
    # budget gate before R2: measured per-set cost from the R1 NFE100 sets,
    # plus one extra set for realistic contention waste
    sys.path.insert(0,"experiments/policy_improvement/tal_followup/uwm")
    from program_spend import spent
    secs=[]
    for n in ("ctl10k_b96n100","qaction10k_b96n100"):
        c=json.load(open(P2%(n,1)))["control"]; secs+=[c[s_]["seconds"] for s_ in S]
    per=float(np.mean(secs))/3600+0.01
    proj=spent()+11*per
    print("R2 budget: spent %.3f + 11 x %.3f (10 sets + 1 waste) = %.3f vs cap 17"%(spent(),per,proj))
    if proj>17.0:
        print("R2 would exceed the cap -- stopping"); sys.exit(10)
    sys.exit(0)

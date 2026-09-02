import json, numpy as np, csv
from itertools import product
exec(open("experiments/topconf/paper_figures/make_figures.py").read().split("# ---------------- FIG 1")[0])
S8,e8=bank(); Y8=arr(S8,e8); w_ref=(Y8.var(0,ddof=1)).mean()
def boot_hw(X,Y,R,nb=4000,seed=0):
    rng=np.random.default_rng(seed); N=X.shape[1]; v=[]
    for _ in range(nb):
        idx=rng.integers(0,N,N)
        ra=rng.integers(0,X.shape[0],(R,N)); rb=rng.integers(0,Y.shape[0],(R,N))
        a=np.mean([X[ra[k],np.arange(N)] for k in range(R)],0)[idx].mean()
        b=np.mean([Y[rb[k],np.arange(N)] for k in range(N and R)],0)[idx].mean()
        v.append(100*(b-a))
    lo,hi=np.percentile(v,[2.5,97.5]); return (hi-lo)/2
f=lambda z:"+" if z>5 else("-" if z<-5 else "0")
with open(f"{OUT}/calibrated_contrasts.csv","w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["task","seed","arm_A_nfe","arm_B_nfe","n_scenarios","R_per_arm","succ_A","succ_B",
                "delta_pp","ci95_lo_pp","ci95_hi_pp","heldout_SE_R1_pp","signal_to_resolution",
                "n_nested_R1_views","sign_disagree_frac","category_disagree_frac","practical_category"])
    for (t,s,a,b) in CON:
        X,Y=ARMS[(t,s,a)],ARMS[(t,s,b)]; cal=100*(Y.mean()-X.mean())
        h=boot_hw(X,Y,3,seed=1000+s)
        ws=[ARMS[k].var(0,ddof=1).mean() for k in ARMS if k[0]==t and k[1]!=s]
        se1=100*np.sqrt(2*np.mean(ws)/(X.shape[1]*1))
        v=[100*(Y[j].mean()-X[i].mean()) for i,j in product(range(3),range(3))]
        w.writerow([t,s,a,b,X.shape[1],3,f"{X.mean():.4f}",f"{Y.mean():.4f}",f"{cal:+.2f}",
                    f"{cal-h:+.2f}",f"{cal+h:+.2f}",f"{se1:.2f}",f"{abs(cal)/se1:.2f}",9,
                    f"{sum(1 for x in v if abs(x)>1e-9 and np.sign(x)!=np.sign(cal))/9:.3f}",
                    f"{sum(1 for x in v if f(x)!=f(cal))/9:.3f}",f(cal)])
print(open(f"{OUT}/calibrated_contrasts.csv").read())

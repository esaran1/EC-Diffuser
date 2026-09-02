import json,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
exec(open("experiments/topconf/paper_figures/make_figures.py").read().split("# ---------------- FIG 1")[0])
L1={k:100*ARMS[k].mean(1) for k in ARMS}
S8,e8=bank(); Y8=arr(S8,e8); w=Y8.var(0,ddof=1).mean()

fig,(axL,axR)=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={"width_ratios":[1,1.25]})

# LEFT: aggregate SD contracts as units are averaged
lv=[]
lv.append(("scenario\n(Level 0)",100*Y8.std(0,ddof=1).mean()))
lv.append(("one arm\n(Level 1)",np.mean([np.std(v,ddof=1) for v in L1.values()])))
l2=[]
for t in ("3c","4c"):
    for n in sorted({k[2] for k in ARMS if k[0]==t}):
        ck=[k for k in ARMS if k[0]==t and k[2]==n]
        if len(ck)==3: l2.append(np.std(np.mean([L1[k] for k in ck],0),ddof=1))
lv.append(("arm x 3 ckpt\n(Level 2)",np.mean(l2)))
lv.append(("task regime\n(Level 3)",np.mean([np.std(np.mean([L1[k] for k in ARMS if k[0]==t],0),ddof=1) for t in ("3c","4c")])))
axL.bar(range(4),[x[1] for x in lv],color=["#999","#7f9fc4","#4a7bb0","#2166ac"],width=.62)
for i,(n,v) in enumerate(lv): axL.text(i,v+0.45,f"{v:.2f}",ha="center",fontsize=7.5,fontweight="bold")
axL.set_xticks(range(4)); axL.set_xticklabels([x[0] for x in lv],fontsize=6.8)
axL.set_ylabel("run-to-run SD (pp)"); axL.set_ylim(0,20)
axL.set_title("Aggregate score looks increasingly stable\nas evaluation units are averaged",fontsize=8)
axL.annotate("looks like a\nhighly reproducible\nbenchmark",xy=(3,0.42),xytext=(1.75,7.5),
             fontsize=6.8,ha="center",color="#2166ac",
             arrowprops=dict(arrowstyle="->",color="#2166ac",lw=.9))

# RIGHT: the same realizations leave contrasts crossing zero
def hw(X,Y,R,nb=3000,seed=0):
    rng=np.random.default_rng(seed); N=X.shape[1]; v=[]
    for _ in range(nb):
        idx=rng.integers(0,N,N)
        ra=rng.integers(0,X.shape[0],(R,N)); rb=rng.integers(0,Y.shape[0],(R,N))
        a=np.mean([X[ra[k],np.arange(N)] for k in range(R)],0)[idx].mean()
        b=np.mean([Y[rb[k],np.arange(N)] for k in range(R)],0)[idx].mean()
        v.append(100*(b-a))
    lo,hi=np.percentile(v,[2.5,97.5]); return (hi-lo)/2
rows=[]
for (t,s,a,b) in CON:
    X,Y=ARMS[(t,s,a)],ARMS[(t,s,b)]
    d=100*(Y.mean()-X.mean()); h=hw(X,Y,3,seed=1000+s)
    v=np.array([100*(Y[j].mean()-X[i].mean()) for i,j in product(range(3),range(3))])
    rows.append((f"{t} s{s}\nNFE{b}-{a}",d,h,v))
rows.sort(key=lambda r:r[1])
y=np.arange(len(rows))
axR.axvspan(-5,5,color="#eeeeee",zorder=0)
axR.axvline(0,color="k",lw=.9,zorder=1)
for i,(lb,d,h,v) in enumerate(rows):
    axR.plot([d-h,d+h],[i,i],color="#2166ac",lw=1.6,zorder=2)
    axR.scatter(v,[i]*len(v),s=9,facecolors="none",edgecolors="#b2182b",lw=.7,zorder=3)
    axR.scatter([d],[i],s=22,color="#2166ac",zorder=4)
axR.set_yticks(y); axR.set_yticklabels([r[0] for r in rows],fontsize=5.8)
axR.set_xlabel("Δ success (pp)")
axR.set_title("…while every constituent contrast from the SAME runs\nstill crosses zero (12/12 CIs include 0)",fontsize=8)
axR.scatter([],[],s=22,color="#2166ac",label="R=3 estimate ±95% CI")
axR.scatter([],[],s=9,facecolors="none",edgecolors="#b2182b",label="single-realization (R=1) views")
axR.legend(fontsize=6,loc="lower right",framealpha=.95)
fig.suptitle("Aggregate stability does not imply comparison resolution",fontsize=10,y=1.02,fontweight="bold")
fig.tight_layout()
fig.savefig("experiments/topconf/paper_figures/fig_aggregate_vs_local_resolution.pdf")
fig.savefig("experiments/topconf/paper_figures/fig_aggregate_vs_local_resolution.png")
print("wrote fig_aggregate_vs_local_resolution")
for n,v in lv: print(f"  {n.split(chr(10))[0]:>14s}: {v:.2f} pp")

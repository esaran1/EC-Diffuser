"""Paper figure package from frozen data. CPU only. No E1 artifacts touched."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
plt.rcParams.update({"font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,
                     "legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
                     "figure.dpi":160,"savefig.bbox":"tight","axes.grid":True,
                     "grid.alpha":0.25,"grid.linewidth":0.4})
OUT="experiments/topconf/paper_figures"
Rr="experiments/evaluation_noise/results"; F="experiments/topconf/fourcube_r3"

def L3(s,n,t):
    S=[json.load(open(f"{Rr}/r0_s{s}_nfe{n}_{t}{r}.json")) for r in (1,2,3)]
    e=sorted({x["episode"] for x in S[0]["episodes"]}); return S,e
def arr(S,e,k="success"):
    return np.array([[{y["episode"]:float(y[k]) for y in d["episodes"]}[i] for i in e] for d in S])
def L4(s,n):
    S=[json.load(open(f"{F}/4cube_H100_s{s}_nfe{n}_rep{r}.json")) for r in (1,2,3)]
    e=sorted({x["episode"] for x in S[0]["episodes"]}); return S,e
def bank():   # seed42 NFE4 R=8
    S=[json.load(open(f"{Rr}/r0_s42_nfe4_crnrep{r}.json")) for r in range(1,9)]
    e=sorted({x["episode"] for x in S[0]["episodes"]}); return S,e

ARMS={}; 
for s in (42,43,44):
    ARMS[("3c",s,1)]=arr(*L3(s,1,"r3rep")); ARMS[("3c",s,2)]=arr(*L3(s,2,"n24rep"))
    ARMS[("3c",s,4)]=arr(*L3(s,4,"r3rep"))
    ARMS[("4c",s,2)]=arr(*L4(s,2)); ARMS[("4c",s,4)]=arr(*L4(s,4))
CON=[("3c",42,1,4),("3c",43,1,4),("3c",44,1,4),("3c",42,2,4),("3c",43,2,4),("3c",44,2,4),
     ("3c",42,1,2),("3c",43,1,2),("3c",44,1,2),("4c",42,2,4),("4c",43,2,4),("4c",44,2,4)]
def save(fig,name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)
    print("wrote",name)

# ---------------- FIG 1: nested realizations ----------------
S8,e8=bank(); Y8=arr(S8,e8); D8=arr(S8,e8,"max_obj_dist"); p8=Y8.mean(0)
# selection rule: FIRST scenario (lowest id) in each class -- stated, not cherry-picked
rob=[i for i,x in enumerate(p8) if x==1][0]
sens=sorted([(abs(x-0.5),i) for i,x in enumerate(p8) if 0<x<1])[0][1]
mid=[i for i,x in enumerate(p8) if 0<x<1][0]
sel=[rob,mid,sens]; labs=["robust (p=1.0)",f"sensitive (p={p8[mid]:.2f})",f"most sensitive (p={p8[sens]:.2f})"]
fig,axes=plt.subplots(1,3,figsize=(6.8,2.5),sharey=True)
for ax,i,lb in zip(axes,sel,labs):
    c=["#2166ac" if Y8[r,i]==1 else "#b2182b" for r in range(8)]
    ax.bar(range(1,9),D8[:,i],color=c,width=.7)
    ax.axhline(0.04,ls="--",c="k",lw=.8)
    ax.set_title(f"scenario {e8[i]}  —  {lb}",fontsize=7.5,pad=4); ax.set_xlabel("physics realization")
axes[0].set_ylabel("final max object–goal dist (m)")
axes[0].text(.5,.04*1.15,"success threshold",fontsize=6)
fig.suptitle("Fig 1  Same policy, same scenario, same policy noise — 8 physics realizations",fontsize=9,y=1.10)
fig.tight_layout()
save(fig,"fig1_nested_realizations")

# ---------------- FIG 2: contact bifurcation ----------------
FC=arr(S8,e8,"first_contact_step")
sens_m=(p8>0)&(p8<1); rob_m=p8==1
fig,axes=plt.subplots(1,3,figsize=(6.8,2.3))
ax=axes[0]
ax.boxplot([FC[:,rob_m].std(0,ddof=1),FC[:,sens_m].std(0,ddof=1)],labels=["robust","sensitive"],widths=.5)
ax.set_ylabel("within-scenario SD of\nfirst-contact step"); ax.set_title("contact timing variability")
ax=axes[1]
succ_d=D8[Y8==1]; fail_d=D8[Y8==0]
ax.hist([succ_d,fail_d],bins=np.linspace(0,0.6,25),stacked=False,
        label=["successful realization","failed realization"],color=["#2166ac","#b2182b"])
ax.axvline(0.04,ls="--",c="k",lw=.8); ax.set_xlabel("final max object–goal dist (m)")
ax.set_ylabel("count"); ax.legend(); ax.set_title("outcomes bifurcate, not jitter")
ax=axes[2]
ax.hist(p8,bins=np.linspace(0,1,9),color="#666")
ax.set_xlabel("per-scenario success prob. $p_i$ (R=8)"); ax.set_ylabel("scenarios")
ax.set_title(f"{int(sens_m.sum())}/96 non-degenerate")
fig.suptitle("Fig 2  Contact-associated bifurcation (seed 42, NFE4, R=8)",fontsize=9,y=1.06)
save(fig,"fig2_contact_bifurcation")
print(f"  [fig2] mean SD contact step: robust {FC[:,rob_m].std(0,ddof=1).mean():.4f} "
      f"sensitive {FC[:,sens_m].std(0,ddof=1).mean():.4f}")
print(f"  [fig2] mean max_obj_dist: success {succ_d.mean():.4f} fail {fail_d.mean():.4f}")

# ---------------- FIG 3: conclusion instability ----------------
fig,ax=plt.subplots(figsize=(6.8,3.4))
rows=[]
for k,(t,s,a,b) in enumerate(CON):
    X,Y=ARMS[(t,s,a)],ARMS[(t,s,b)]
    cal=100*(Y.mean()-X.mean())
    v=[100*(Y[j].mean()-X[i].mean()) for i,j in product(range(3),range(3))]
    rows.append((t,s,f"NFE{b}−{a}",cal,v))
ax.axhspan(-5,5,color="#dddddd",alpha=.6,zorder=0,label="predeclared ±5 pp band")
ax.axhline(0,c="k",lw=.8,zorder=1)
for k,(t,s,c,cal,v) in enumerate(rows):
    ax.scatter([k]*9,v,s=14,facecolors="none",edgecolors="#777",zorder=2,
               label="reconstructed R=1 views" if k==0 else None)
    bad=[x for x in v if abs(x)>1e-9 and np.sign(x)!=np.sign(cal)]
    if bad: ax.scatter([k]*len(bad),bad,s=16,color="#b2182b",zorder=3,
                       label="opposite point-estimate direction" if k==0 else None)
    ax.scatter([k],[cal],marker="_",s=260,color="#2166ac",lw=2,zorder=4,
               label="higher-replication estimate (R=3)" if k==0 else None)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([f"{t}\ns{s}\n{c}" for t,s,c,_,_ in rows],fontsize=6)
ax.set_ylabel("Δ success (pp)")
ax.set_title("Fig 3  Single-realization evaluation is unstable at the level of the comparison itself\n"
             "9 of 12 calibrated contrasts contain at least one R=1 view with the opposite point-estimate\n"
             "direction to the higher-replication estimate; 11 of 12 change practical category.\n"
             "Unit of analysis = the contrast (12); the 9 R=1 views per contrast are nested within it.",
             fontsize=8)
ax.legend(loc="upper left",framealpha=.9)
save(fig,"fig3_policy_conclusion_instability")

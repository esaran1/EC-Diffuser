"""Slide-quality visuals for weekly research meeting. Frozen artifacts only.
Sources:
  A: experiments/loss_balance_audit/ARM_NEUTRAL_NFE_CURVE.md (3-seed means)
  B: experiments/topconf/lerobot_pusht/NFE2_VS_PUBLISHED_NFE100_N500.md
  C: experiments/topconf/paper_figures/contrast_level_instability.csv
  D: experiments/topconf/final_hardening/RESOLUTION_AUDIT_FEASIBILITY.md
  E: reuse of make_fig_masking.py numbers
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size":17,"axes.titlesize":21,"axes.labelsize":19,
    "xtick.labelsize":17,"ytick.labelsize":17,"legend.fontsize":15,
    "axes.grid":True,"grid.alpha":0.22,"grid.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":1.4,"savefig.bbox":"tight","figure.dpi":200,
})
OUT="experiments/topconf/weekly_slides_visuals"
BLUE,RED,GREY="#1f6bb0","#c0392b","#7f8c8d"
def save(fig,n):
    fig.savefig(f"{OUT}/{n}.pdf"); fig.savefig(f"{OUT}/{n}.png",dpi=200)
    fig.savefig(f"{OUT}/{n}.svg"); plt.close(fig); print("wrote",n)

# ---------------- A: action vs state inference depth ----------------
nfe=np.array([1,2,4,8,16,32,512])
act=np.array([0.01785,0.01717,0.02212,0.03353,0.03824,0.04136,0.04592])
sta=np.array([0.21032,0.10761,0.04881,0.02960,0.02566,0.02493,0.02566])
fig,(a1,a2)=plt.subplots(2,1,figsize=(9,8),sharex=True)
a1.plot(nfe,act,"o-",color=BLUE,lw=3.2,ms=11)
a1.scatter([2],[act[1]],s=300,facecolors="none",edgecolors=RED,lw=3.5,zorder=5)
a1.annotate("minimum\nat NFE 2",xy=(2,act[1]),xytext=(5,0.0195),color=RED,fontsize=17,
            arrowprops=dict(arrowstyle="->",color=RED,lw=2.2))
a1.set_ylabel("action error"); a1.set_title("Action and state prefer different inference depths",pad=14)
a2.plot(nfe,sta,"s-",color="#117a3d",lw=3.2,ms=11)
a2.axvspan(16,32,color="#117a3d",alpha=.13,zorder=0)
a2.annotate("plateau\nNFE 16–32",xy=(22,0.0256),xytext=(45,0.10),color="#117a3d",fontsize=17,
            arrowprops=dict(arrowstyle="->",color="#117a3d",lw=2.2))
a2.set_ylabel("state error"); a2.set_xlabel("NFE  (model calls per plan)")
for ax in (a1,a2):
    ax.set_xscale("log",base=2); ax.set_xticks(nfe)
    ax.set_xticklabels([str(x) for x in nfe]); ax.minorticks_off()
fig.tight_layout(); save(fig,"A_action_vs_state_depth")

# ---------------- B: PushT compute vs behavior ----------------
fig,ax=plt.subplots(figsize=(9,5.6))
ax.plot([18.9,702.8],[60.8,65.4],"-",color=GREY,lw=2.5,zorder=1)
ax.scatter([18.9],[60.8],s=420,color=BLUE,zorder=3)
ax.scatter([702.8],[65.4],s=420,color=RED,zorder=3)
ax.annotate("NFE 2",xy=(18.9,60.8),xytext=(21,58.6),fontsize=20,color=BLUE,fontweight="bold")
ax.annotate("NFE 100",xy=(702.8,65.4),xytext=(330,66.6),fontsize=20,color=RED,fontweight="bold")
ax.annotate("",xy=(702.8,62.6),xytext=(18.9,62.6),
            arrowprops=dict(arrowstyle="<->",color="#333",lw=2.4))
ax.text(115,63.1,"37× latency",fontsize=19,ha="center",fontweight="bold")
ax.text(115,61.2,"+4.6 pp success",fontsize=19,ha="center")
ax.set_xscale("log"); ax.set_xticks([20,50,100,200,500,1000])
ax.set_xticklabels(["20","50","100","200","500","1000"]); ax.minorticks_off()
ax.set_xlabel("planner latency (ms, log scale)"); ax.set_ylabel("PushT success (%)")
ax.set_ylim(57,68.5); ax.set_title("PushT: 37× more compute buys 4.6 pp  (n = 500)",pad=14)
ax.text(0.5,-0.30,"offline action L2 improves ~21%",transform=ax.transAxes,
        fontsize=15,color=GREY,ha="center")
fig.tight_layout(); save(fig,"B_pusht_compute_vs_behavior")

# ---------------- C: single-realization comparison instability ----------------
rows=list(csv.DictReader(open("experiments/topconf/paper_figures/contrast_level_instability.csv")))
import json
exec(open("experiments/topconf/paper_figures/make_figures.py").read().split("# ---------------- FIG 1")[0])
OUT="experiments/topconf/weekly_slides_visuals"   # exec above clobbers OUT; restore
from itertools import product
views={}
for (t,s,a,b) in CON:
    X,Y=ARMS[(t,s,a)],ARMS[(t,s,b)]
    views[(("3-cube" if t=="3c" else "4-cube"),str(s),f"NFE{b}-NFE{a}")]=\
        [100*(Y[j].mean()-X[i].mean()) for i,j in product(range(3),range(3))]
rows.sort(key=lambda r:float(r["dR"]))
fig,ax=plt.subplots(figsize=(9.6,7.4))
ax.axvspan(-5,5,color="#ececec",zorder=0)
ax.axvline(0,color="k",lw=2.2,zorder=1)
for i,r in enumerate(rows):
    v=views[(r["task"],r["ckpt"],r["con"])]
    both=r["both"]=="True"
    ax.scatter(v,[i]*len(v),s=70,facecolors="none",
               edgecolors=RED if both else GREY,lw=2.0,zorder=3)
    ax.scatter([float(r["dR"])],[i],s=230,color=BLUE,zorder=4,marker="D")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['task'][0]}c s{r['ckpt']}  {r['con'].replace('NFE','')}" for r in rows],fontsize=15)
ax.set_xlabel("Δ success  (percentage points)",fontsize=19)
ax.set_title("9 / 12 contrasts admit both effect directions under R = 1",pad=14,fontsize=21)
ax.scatter([],[],s=230,color=BLUE,marker="D",label="higher-replication estimate")
ax.scatter([],[],s=70,facecolors="none",edgecolors=RED,lw=2.0,label="single-realization (R=1) views")
ax.legend(loc="lower right",framealpha=.96,fontsize=16)
ax.text(0.5,-0.145,"median within-contrast R=1 spread: 10.4 pp",transform=ax.transAxes,
        fontsize=17,ha="center",color="#333")
fig.tight_layout(); save(fig,"C_single_realization_instability")

# ---------------- D: resolution vs physics repeats ----------------
R=np.array([1,3,5,8]); hw=np.array([7.55,4.36,3.38,2.67])   # frozen audit
fig,ax=plt.subplots(figsize=(8.4,6.4))
for yv in (10,5,3):
    ax.axhline(yv,ls="--",color="#c4c4c4",lw=1.8,zorder=0)
    ax.text(0.62,yv+0.28,f"{yv} pp",fontsize=16,color="#8a8a8a",va="bottom",ha="left")
ax.plot(R,hw,"o-",color=BLUE,lw=4.0,ms=17,zorder=3)
for x,y in zip(R,hw):
    ax.annotate(f"{y:.1f}",xy=(x,y),xytext=(x+0.22,y+0.62),fontsize=21,
                ha="left",fontweight="bold",color=BLUE)
ax.set_xticks(R); ax.set_xlim(0.4,9.0); ax.set_ylim(0,11.2)
ax.set_xlabel("physics repetitions per scenario  (R)",fontsize=20)
ax.set_ylabel("95% effect half-width (pp)",fontsize=20)
ax.set_title("What effect size can the evaluator resolve?",pad=16,fontsize=22)
ax.tick_params(labelsize=19)
fig.tight_layout(); save(fig,"D_resolution_vs_repeats")

# ---------------- E: aggregate stability vs local resolution ----------------
W=[float(r["width"]) for r in rows]
fig,ax=plt.subplots(figsize=(9.2,6.0))
vals=[0.64,0.20,float(np.median(W))]
cols=[BLUE,BLUE,RED]
ax.bar([0,1,2.35],vals,color=cols,width=.62)
for x,v in zip([0,1,2.35],vals):
    ax.text(x,v+0.32,f"{v:.2f} pp" if v<1 else f"{v:.1f} pp",ha="center",
            fontsize=23,fontweight="bold",color=BLUE if v<1 else RED)
ax.set_xticks([0,1,2.35])
ax.set_xticklabels(["3-cube\naggregate SD","4-cube\naggregate SD","median contrast\nR=1 range"],fontsize=18)
ax.set_ylabel("percentage points",fontsize=20); ax.set_ylim(0,12.6)
ax.tick_params(axis="y",labelsize=19)
ax.text(0.5,11.4,"same evaluation data",ha="center",fontsize=17,color="#666")
ax.annotate("",xy=(2.05,11.0),xytext=(1.3,11.0),
            arrowprops=dict(arrowstyle="->",color="#666",lw=2.2))
ax.set_title("Aggregate stability  \u2260  comparison resolution",
             fontsize=24,fontweight="bold",pad=16)
ax.text(2.35,-2.55,"9/12 contrasts span both directions",ha="center",
        fontsize=17,color=RED)
ax.text(0.5,-2.55,"SD and range are different statistics",ha="center",
        fontsize=14,color="#888",style="italic")
fig.tight_layout(); save(fig,"E_aggregate_vs_local")

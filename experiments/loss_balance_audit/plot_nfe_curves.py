"""Two curves vs NFE: control performance and imagination error. Values ingested
from canonical JSON; nothing typed in."""
import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NFES=[1,2,4,8,16]
S=json.load(open("experiments/loss_balance_audit/nfe_imagination_sweep.json"))["results"]
img={n:[np.mean(S[f"flow_s{s}_nfe{n}"]["chamfer"]) for s in (42,43,44)] for n in NFES}
gauss=np.mean(S["gaussian_nfe100"]["chamfer"]); copy=np.mean(S["flow_s42_copy"]["chamfer"])

ctrl={}
for n in NFES:
    placed=[]
    for f in glob.glob(f"experiments/isaacgym_control/nfe_study/r*_flow_nfe{n}.json"):
        d=json.load(open(f)); placed += [e["cubes_placed"] for e in d["episodes"]]
    ctrl[n]=np.mean(placed)/3.0

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.8))

m=[np.mean(img[n]) for n in NFES]; sd=[np.std(img[n]) for n in NFES]
ax1.errorbar(NFES,m,yerr=sd,fmt="o-",color="#1f77b4",lw=2.5,ms=9,capsize=4,
             label="Flow imagination error (3 seeds)")
ax1.axhline(gauss,ls="--",c="#d62728",lw=2,label=f"Gaussian@100 = {gauss:.4f}")
ax1.axhline(copy,ls=":",c="grey",lw=2,label=f"copy current state = {copy:.4f}")
ax1.set_xscale("log",base=2); ax1.set_xticks(NFES); ax1.set_xticklabels(NFES)
ax1.set_xlabel("Flow NFE (Euler steps)"); ax1.set_ylabel("chamfer, position (lower better)")
ax1.set_title("Future-state prediction error vs NFE",fontsize=12)
ax1.grid(alpha=.3); ax1.legend(fontsize=8.5)
ax1.annotate(f"residual gap\n{np.mean(img[16])-gauss:+.4f} ({np.mean(img[16])/gauss:.2f}x)",
             xy=(16,np.mean(img[16])),xytext=(4.2,0.105),fontsize=9,color="#b22222",
             arrowprops=dict(arrowstyle="->",color="#b22222"))

axb=ax2.twinx()
ax2.plot(NFES,[ctrl[n] for n in NFES],"s-",color="#2ca02c",lw=2.5,ms=9,label="control (seed 42)")
axb.plot(NFES,m,"o--",color="#1f77b4",lw=2,ms=7,label="imagination (3 seeds)")
ax2.set_xscale("log",base=2); ax2.set_xticks(NFES); ax2.set_xticklabels(NFES)
ax2.set_xlabel("Flow NFE"); ax2.set_ylabel("per-object success (higher better)",color="#2ca02c")
axb.set_ylabel("chamfer error (lower better)",color="#1f77b4")
ax2.set_title("Control saturates early; state prediction does not",fontsize=12)
ax2.grid(alpha=.3)
h1,l1=ax2.get_legend_handles_labels(); h2,l2=axb.get_legend_handles_labels()
ax2.legend(h1+h2,l1+l2,fontsize=8.5,loc="center right")
ax2.annotate("control ~saturated by NFE2",xy=(2,ctrl[2]),xytext=(2.4,0.912),fontsize=9,
             color="#2ca02c",arrowprops=dict(arrowstyle="->",color="#2ca02c"))

plt.suptitle("Same trained weights, same initial noise — only Euler discretisation changes",fontsize=13)
plt.tight_layout()
os.makedirs("experiments/figures",exist_ok=True)
p="experiments/figures/nfe_imagination_vs_control.png"
plt.savefig(p,dpi=150); print("wrote",p)

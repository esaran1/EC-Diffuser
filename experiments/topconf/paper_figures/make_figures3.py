"""Fig 4 + Fig 5 rebuilt against the FROZEN Resolution Audit protocol."""
import json, numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
exec(open("experiments/topconf/paper_figures/make_figures.py").read().split("# ---------------- FIG 1")[0])

S8,e8=bank(); Y8=arr(S8,e8); N=Y8.shape[1]
w_ref=(Y8.var(0,ddof=1)).mean()          # bias-corrected within-scenario variance
b_var=Y8.mean(0).var(ddof=1)             # between-scenario
print(f"w_ref={w_ref:.5f}  between={b_var:.5f}")

def se_delta(w,N,R):  # two independent arms, Cov=0 by design
    return 100*np.sqrt(2*w/(N*R))

# ---------- FIG 4: analytic resolution + frozen audit anchors ----------
Rs=np.arange(1,13)
se=[se_delta(w_ref,N,R) for R in Rs]; hw=[1.96*x for x in se]
frozen={1:7.55,3:4.36,5:3.38,8:2.67}
fig,axes=plt.subplots(1,2,figsize=(6.8,2.6))
ax=axes[0]
ax.plot(Rs,hw,"-",color="#2166ac",label="analytic 95% half-width")
ax.plot(list(frozen),list(frozen.values()),"o",color="#b2182b",label="frozen audit (bootstrap)")
ax.axhline(5,ls="--",c="k",lw=.9)
ax.text(6.2,5.25,"predeclared 5 pp band",fontsize=6)
ax.set_xlabel("realizations per scenario, R"); ax.set_ylabel("resolution: 95% CI half-width (pp)")
ax.legend(fontsize=6); ax.set_title(f"N={N} scenarios per arm")
ax=axes[1]
for NN,c in [(96,"#2166ac"),(250,"#66a61e"),(500,"#e6ab02")]:
    ax.plot(Rs,[1.96*se_delta(w_ref,NN,R) for R in Rs],"-",color=c,label=f"N={NN}")
ax.axhline(5,ls="--",c="k",lw=.9)
ax.set_xlabel("realizations per scenario, R"); ax.set_ylabel("95% CI half-width (pp)")
ax.legend(fontsize=6); ax.set_title("scenarios vs replication trade-off")
fig.suptitle("Fig 4  What the evaluator can actually resolve",fontsize=9,y=1.05)
save(fig,"fig4_resolution_calibration")
for R in [1,2,3,5,8]: print(f"  [fig4] R={R}: analytic hw={1.96*se_delta(w_ref,N,R):.2f}pp"
                            + (f"  frozen={frozen[R]}" if R in frozen else ""))

# ---------- FIG 5: frozen held-out protocol, binned by signal-to-resolution ----------
def heldout_se(t,s,R=1):
    """w from the OTHER seeds of the same task only -> genuinely held out"""
    ws=[]
    for (tt,ss,aa) in ARMS:
        if tt==t and ss!=s: ws.append(ARMS[(tt,ss,aa)].var(0,ddof=1).mean())
    return 100*np.sqrt(2*np.mean(ws)/(ARMS[(t,s,list({a for (tt,ss,a) in ARMS if tt==t})[0])].shape[1]*R))
rows=[]
for (t,s,a,b) in CON:
    X,Y=ARMS[(t,s,a)],ARMS[(t,s,b)]; cal=100*(Y.mean()-X.mean())
    se1=heldout_se(t,s,1); ratio=abs(cal)/se1
    v=[100*(Y[j].mean()-X[i].mean()) for i,j in product(range(3),range(3))]
    f=lambda z:"+" if z>5 else("-" if z<-5 else "0")
    rows.append((ratio,sum(1 for x in v if abs(x)<=1e-9 or np.sign(x)==np.sign(cal)),
                 sum(1 for x in v if f(x)==f(cal)),9))
bins=[(0,.25),(.25,.5),(.5,1.),(1.,9e9)]; lab=["[0,0.25)","[0.25,0.5)","[0.5,1)","[1,∞)"]
sg=[];ct=[];nn=[]
for lo,hi in bins:
    g=[r for r in rows if lo<=r[0]<hi]
    tot=sum(r[3] for r in g) or 1
    sg.append(100*sum(r[1] for r in g)/tot); ct.append(100*sum(r[2] for r in g)/tot); nn.append(len(g))
fig,ax=plt.subplots(figsize=(4.6,2.9))
x=np.arange(4)
ax.bar(x-.19,sg,.38,color="#2166ac",label="sign of Δ reproduced")
ax.bar(x+.19,ct,.38,color="#b2182b",label="practical category (±5 pp) reproduced")
ax.axhline(50,ls=":",c="k",lw=.8); ax.text(3.15,51,"chance (sign)",fontsize=6)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n({n} contr.)" for l,n in zip(lab,nn)],fontsize=6.5)
ax.set_xlabel("held-out signal-to-resolution ratio  |Δ| / SE(Δ) at R=1")
ax.set_ylabel("% of 9 nested R=1 views\nagreeing with R=3 reference")
ax.set_ylim(0,105); ax.legend(fontsize=6.5,loc="lower right")
ax.set_title("Fig 5  Calibration predicts sign reliability,\nbut NOT practical-category reliability",fontsize=8.5)
save(fig,"fig5_heldout_calibration")
for l,a,b,n in zip(lab,sg,ct,nn): print(f"  [fig5] ratio {l}: n={n} sign={a:.1f}% cat={b:.1f}%")

"""State-first / action-second offline screen (GPU, replay, frozen checkpoints).
64 replay windows x 3 noise seeds; identical source noise across variants.
Convention: t=0 noise, t=1 clean. Every phase is an explicit axis-aligned
Euler integration with fail-loud invariants and exact call counting."""
import sys, json, time, numpy as np, torch
sys.path.insert(0,"diffuser")
from diffuser.utils.serialization import load_config
R="3C_dlp_adalnpint_randcolor_H5_T4_seed42"; AD=3; dev="cuda"; SEEDS=(1,2,3)
def load(pre):
    mc=load_config("data/panda_push/"+pre,R,"model_config.pkl"); dc=load_config("data/panda_push/"+pre,R,"diffusion_config.pkl")
    w=dc._class(mc._class(**mc._dict),**dc._dict).to(dev); w.load_state_dict(torch.load("data/panda_push/%s/%s/state_final.pt"%(pre,R),map_location=dev)["ema"]); return w.eval()
dsc=load_config("data/panda_push/uwm_ctl",R,"dataset_config.pkl"); dsc._dict["dataset_path"]="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"; ds=dsc(seed=42)
idx=np.random.RandomState(20261023).choice(len(ds),64,replace=False); B=64
X1=torch.stack([torch.as_tensor(ds[int(i)].trajectories) for i in idx]).float().to(dev)
C={k:torch.stack([torch.as_tensor(ds[int(i)].conditions[k]) for i in idx]).float().to(dev) for k in ds[int(idx[0])].conditions}
def noise(sd): g=torch.Generator(device="cpu").manual_seed(sd); return torch.randn(X1.shape,generator=g).to(dev)
OUT={}; CALLS=[0]
def prep(w):
    x1=X1.clone(); w._apply_conditioning(x1,C); m=w._make_conditioning_mask(x1,C).float(); return x1,m
def errs(w,x,x1,m):
    om=m[:,:,AD:]
    return {"act0":(x[:,0,:AD]-x1[:,0,:AD]).abs().mean().item(), "act_all":(x[...,:AD]-x1[...,:AD]).abs().mean().item(),
            "state":(((x[...,AD:]-x1[...,AD:]).abs()*om).sum()/om.sum()).item()}
def agg(rs): return {k:float(np.mean([r[k] for r in rs])) for k in rs[0]}

# ---------------- UWM two-time model ----------------
u=load("uwm_uwm_ext"); x1u,mu=prep(u)
def V(x,ts,ta):
    CALLS[0]+=1
    return u._call_two_time_model(x,C,torch.full((B,),ts,device=dev),torch.full((B,),ta,device=dev))
@torch.no_grad()
def state_phase(x,S,act_src=None):
    """t_s: 0->1 over S steps at t_a=0. Updates STATE ONLY."""
    x=x.clone()
    if act_src is not None: x[...,:AD]=act_src
    a0=x[...,:AD].clone()
    for k in range(S):
        v=V(x,k/S,0.0)*mu; x[...,AD:]+= v[...,AD:]/S; u._apply_conditioning(x,C)
    assert torch.equal(x[...,:AD],a0), "state phase touched the action"
    return x
@torch.no_grad()
def action_phase(x,A,a_init):
    """t_a: 0->1 over A steps at t_s=1. Updates ACTION ONLY; state frozen."""
    x=x.clone(); x[...,:AD]=a_init; s0=x[...,AD:].clone()
    for k in range(A):
        v=V(x,1.0,k/A)*mu; x[...,:AD]+= v[...,:AD]/A
    assert torch.equal(x[...,AD:],s0), "action phase touched the state"
    return x
@torch.no_grad()
def joint(x,N):
    x=x.clone()
    for k in range(N):
        v=V(x,k/N,k/N)*mu; x+= v/N; u._apply_conditioning(x,C)
    return x
# sanity: phase invariants + call counting
CALLS[0]=0; z=noise(1); u._apply_conditioning(z,C); _=action_phase(state_phase(z,4),4,z[...,:AD]); assert CALLS[0]==8

res={}
for S,A in ((4,4),(100,4),(96,4)):
    rs=[]
    for sd in SEEDS:
        z=noise(sd); u._apply_conditioning(z,C)
        rs.append(errs(u,action_phase(state_phase(z,S),A,z[...,:AD]),x1u,mu))
    res["S%d_A%d"%(S,A)]=agg(rs)
for N in (4,100):
    rs=[]
    for sd in SEEDS:
        z=noise(sd); u._apply_conditioning(z,C); rs.append(errs(u,joint(z,N),x1u,mu))
    res["joint%d"%N]=agg(rs)
# action-NFE sweep on CACHED S100 plans; plus oracle true state and S4 plan
plans={sd:None for sd in SEEDS}
for sd in SEEDS:
    z=noise(sd); u._apply_conditioning(z,C); plans[sd]=(state_phase(z,100),state_phase(z,4),z)
for A in (1,2,4,10,25,50):
    for lab,src in (("S100",0),("S4",1),("ORACLE",None)):
        rs=[]
        for sd in SEEDS:
            p100,p4,z=plans[sd]
            if src is None:
                base=x1u.clone()
            else:
                base=(p100,p4)[src]
            rs.append(errs(u,action_phase(base,A,z[...,:AD]),x1u,mu))
        res["%s_A%d"%(lab,A)]=agg(rs)
# contamination: state phase with action input = noise vs zeros vs a DIFFERENT noise draw
for lab,f in (("noise",lambda z:z[...,:AD]),("zeros",lambda z:torch.zeros_like(z[...,:AD])),("othernoise",lambda z:noise(999)[...,:AD])):
    rs=[]
    for sd in SEEDS:
        z=noise(sd); u._apply_conditioning(z,C); rs.append(errs(u,state_phase(z,100,f(z)),x1u,mu)["state"])
    res["Sphase100_state_actsrc_"+lab]=float(np.mean(rs))
OUT["uwm"]=res

# ---------------- direct action (query token) ----------------
q=load("qaction_ext"); x1q,mq=prep(q); om=mq[:,:,AD:]
@torch.no_grad()
def qrun(z,N,keep_trace=False):
    x=z.clone(); tr=[]
    for k in range(N):
        out=q._call_model(x,C,torch.full((B,),k/N,device=dev)*q.time_scale); a=out[...,:AD]
        if keep_trace: tr.append((k/N,(a[:,0]-x1q[:,0,:AD]).abs().mean().item()))
        x[...,AD:]+= (out*mq)[...,AD:]/N; q._apply_conditioning(x,C)
    return x,a,tr
@torch.no_grad()
def readout(xs,t): return q._call_model(xs.clone(),C,torch.full((B,),t,device=dev)*q.time_scale)[...,:AD]
def a0(a): return (a[:,0]-x1q[:,0,:AD]).abs().mean().item()
qr={}; finals={4:[],100:[]}
for N in (4,100):
    A_=[];S_=[]
    for sd in SEEDS:
        z=noise(sd); q._apply_conditioning(z,C); xf,a,tr=qrun(z,N,keep_trace=(N==100 and sd==1))
        A_.append(a0(a)); S_.append((((xf[...,AD:]-x1q[...,AD:]).abs()*om).sum()/om.sum()).item()); finals[N].append(xf)
        if tr: qr["trace100"]=[tr[k] for k in (0,10,25,50,75,90,95,99)]
    qr["native_nfe%d"%N]={"act0":float(np.mean(A_)),"state_final":float(np.mean(S_))}
for t in (0.0,0.5,0.75,0.9,0.99):
    qr["readout_t%.2f"%t]={"S4":float(np.mean([a0(readout(x,t)) for x in finals[4]])),
                           "S100":float(np.mean([a0(readout(x,t)) for x in finals[100]])),
                           "TRUE":a0(readout(x1q,t))}
OUT["qaction"]=qr
json.dump(OUT,open("experiments/policy_improvement/tal_followup/statefirst/offline.json","w"),indent=1)
r=OUT["uwm"]
print("UWM10k (act@0 / act_all / state):")
for k in ("joint4","joint100","S4_A4","S100_A4","S96_A4"): print("  %-9s %.4f  %.4f  %.4f"%(k,r[k]["act0"],r[k]["act_all"],r[k]["state"]))
print("  action-phase sweep act@0:     A1     A2     A4     A10    A25    A50")
for lab in ("S4","S100","ORACLE"): print("    %-7s"%lab+"".join(" %.4f"%r["%s_A%d"%(lab,A)]["act0"] for A in (1,2,4,10,25,50)))
print("  state-phase (S100) state err by action input: noise %.4f  zeros %.4f  other-noise %.4f"%tuple(r["Sphase100_state_actsrc_"+l] for l in ("noise","zeros","othernoise")))
print("DIRECT ACTION:"); print("  native NFE4 ",qr["native_nfe4"]); print("  native NFE100",qr["native_nfe100"]); print("  trace100 (t, act@0):",[(round(a,2),round(b,4)) for a,b in qr["trace100"]])
for t in (0.0,0.5,0.75,0.9,0.99): print("  identical readout t=%.2f  S4 %.4f  S100 %.4f  TRUE %.4f"%(t,*[qr["readout_t%.2f"%t][k] for k in ("S4","S100","TRUE")]))

"""Direct-action mechanism (CPU, replay). Same 64 windows, same 3 noise seeds.
(1) native sampler at NFE 4/100: action-at-step-0 error of the action it returns.
(2) readout of the action at every step's t along the NFE-100 path.
(3) IDENTICAL readout on finalized states: S4-final, S100-final, true state,
    each read at the same t values (0.75, 0.99)."""
import sys, json, numpy as np, torch
sys.path.insert(0,"diffuser")
from diffuser.utils.arrays import set_global_device; set_global_device("cpu")
from diffuser.utils.serialization import load_config
torch.set_num_threads(24); R="3C_dlp_adalnpint_randcolor_H5_T4_seed42"; AD=3
def load(pre):
    mc=load_config("data/panda_push/"+pre,R,"model_config.pkl"); dc=load_config("data/panda_push/"+pre,R,"diffusion_config.pkl")
    w=dc._class(mc._class(**mc._dict),**dc._dict); w.load_state_dict(torch.load("data/panda_push/%s/%s/state_final.pt"%(pre,R),map_location="cpu")["ema"]); return w.eval()
dsc=load_config("data/panda_push/uwm_ctl",R,"dataset_config.pkl"); dsc._dict["dataset_path"]="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"; ds=dsc(seed=42)
idx=np.random.RandomState(20261023).choice(len(ds),64,replace=False)
X1=torch.stack([torch.as_tensor(ds[int(i)].trajectories) for i in idx]).float()
C={k:torch.stack([torch.as_tensor(ds[int(i)].conditions[k]) for i in idx]).float() for k in ds[int(idx[0])].conditions}
w=load("qaction_ext"); x1=X1.clone(); w._apply_conditioning(x1,C); mask=w._make_conditioning_mask(x1,C).float(); B=64
om=mask[:,:,AD:]
def serr(x): return (((x[:,:,AD:]-x1[:,:,AD:]).abs()*om).sum()/om.sum()).item()
def aerr(a): return (a[:,0]-x1[:,0,:AD]).abs().mean().item()
@torch.no_grad()
def run(N,seed,trace=False):
    g=torch.Generator().manual_seed(seed); x=torch.randn(x1.shape,generator=g); w._apply_conditioning(x,C)
    tr=[]
    for k in range(N):
        t=torch.full((B,),k/N); out=w._call_model(x,C,t*w.time_scale)
        a=out[:,:,:AD]
        if trace: tr.append((k/N,aerr(a),serr(x)))
        x[...,AD:]=x[...,AD:]+(1.0/N)*(out*mask)[...,AD:]; w._apply_conditioning(x,C)
    return x,a,tr
@torch.no_grad()
def readout(xs,t):
    return w._call_model(xs.clone(),C,torch.full((B,),t)*w.time_scale)[:,:,:AD]
res={}
for N in (4,100):
    A=[];S=[];XF=[]
    for sd in (1,2,3):
        xf,a,tr=run(N,sd,trace=(N==100 and sd==1)); A.append(aerr(a)); S.append(serr(xf)); XF.append(xf)
        if tr: res["trace100"]=tr
    res["native_nfe%d"%N]={"act0":float(np.mean(A)),"state_final":float(np.mean(S))}; res["xf%d"%N]=XF
print("native sampler   NFE4  act@0 %.4f  final-state err %.4f"%(res["native_nfe4"]["act0"],res["native_nfe4"]["state_final"]))
print("native sampler   NFE100 act@0 %.4f  final-state err %.4f"%(res["native_nfe100"]["act0"],res["native_nfe100"]["state_final"]))
print("\nreadout along the NFE-100 path (t, act@0 err, current state err):")
for k in (0,10,25,50,75,90,95,99):
    t,a,s=res["trace100"][k]; print("   t=%.2f  act %.4f  state %.4f"%(t,a,s))
print("\nIDENTICAL readout on finalized states (act@0 err, mean of 3 seeds):")
xt=x1.clone()
for t in (0.75,0.99):
    r4=np.mean([aerr(readout(x,t)) for x in res["xf4"]]); r100=np.mean([aerr(readout(x,t)) for x in res["xf100"]]); rt=aerr(readout(xt,t))
    print("   readout t=%.2f : S4-final %.4f | S100-final %.4f | TRUE state %.4f"%(t,r4,r100,rt))
del res["xf4"],res["xf100"]
json.dump(res,open("experiments/policy_improvement/tal_followup/statefirst/qaction_readout.json","w"),indent=1)

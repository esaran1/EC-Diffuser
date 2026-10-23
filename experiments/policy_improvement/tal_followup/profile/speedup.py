import sys, time, json, copy, torch
sys.path.insert(0,"diffuser")
from diffuser.utils.serialization import load_config
R="3C_dlp_adalnpint_randcolor_H5_T4_seed42"; pre="uwm_ctl_ext"
mc=load_config("data/panda_push/"+pre,R,"model_config.pkl"); dc=load_config("data/panda_push/"+pre,R,"diffusion_config.pkl")
w=dc._class(mc._class(**mc._dict),**dc._dict).cuda(); w.load_state_dict(torch.load("data/panda_push/%s/%s/state_final.pt"%(pre,R),map_location="cuda")["ema"]); w.eval()
m=w.model
def setsdpa(mod,on):
    for b in mod.particle_transformer.blocks: b.attn.use_sdpa=on
def bench(f,x,t,n=50):
    with torch.no_grad():
        for _ in range(10): f(x,None,t)
        torch.cuda.synchronize(); s=time.time()
        for _ in range(n): f(x,None,t)
        torch.cuda.synchronize()
    return (time.time()-s)/n*1000
res={}
for B in (16,96):
    torch.manual_seed(0); x=torch.randn(B,5,483,device="cuda"); t=torch.full((B,),500.,device="cuda")
    setsdpa(m,False)
    with torch.no_grad(): ref=m(x,None,t).float()
    r={"baseline":(bench(m,x,t),0.0)}
    setsdpa(m,True)
    with torch.no_grad(): o=m(x,None,t).float()
    r["sdpa"]=(bench(m,x,t),(o-ref).abs().max().item())
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    with torch.no_grad(): o=m(x,None,t).float()
    r["sdpa+tf32"]=(bench(m,x,t),(o-ref).abs().max().item())
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    def bf(x_,c_,t_):
        with torch.autocast("cuda",dtype=torch.bfloat16): return m(x_,c_,t_)
    with torch.no_grad(): o=bf(x,None,t).float()
    r["sdpa+bf16"]=(bench(bf,x,t),(o-ref).abs().max().item())
    def fp16(x_,c_,t_):
        with torch.autocast("cuda",dtype=torch.float16): return m(x_,c_,t_)
    with torch.no_grad(): o=fp16(x,None,t).float()
    r["sdpa+fp16"]=(bench(fp16,x,t),(o-ref).abs().max().item())
    try:
        mcmp=torch.compile(m)
        with torch.no_grad(): o=mcmp(x,None,t).float()
        r["sdpa+compile(fp32)"]=(bench(mcmp,x,t),(o-ref).abs().max().item())
    except Exception as e:
        r["sdpa+compile(fp32)"]=(None,str(e)[:80])
    res[B]=r
    print("\nbatch %d   ms/forward   max|d| vs baseline (output scale ~1)"%B)
    for k,(ms,d) in r.items(): print("   %-20s %s   %s"%(k, "%.2f"%ms if ms else "n/a", "%.2e"%d if isinstance(d,float) else d))
json.dump(res,open("experiments/policy_improvement/tal_followup/profile/speedup.json","w"),indent=1)

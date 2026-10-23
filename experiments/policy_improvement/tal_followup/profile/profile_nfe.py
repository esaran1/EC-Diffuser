"""Per-NFE cost of one EC-Diffuser forward (ctl10k EMA), CUDA-synchronized.
Module-level CUDA timing via forward hooks + torch.profiler kernel breakdown."""
import sys, time, json, torch
from collections import defaultdict
sys.path.insert(0,"diffuser")
from diffuser.utils.serialization import load_config
R="3C_dlp_adalnpint_randcolor_H5_T4_seed42"; pre="uwm_ctl_ext"
mc=load_config("data/panda_push/"+pre,R,"model_config.pkl"); dc=load_config("data/panda_push/"+pre,R,"diffusion_config.pkl")
w=dc._class(mc._class(**mc._dict),**dc._dict).cuda(); w.load_state_dict(torch.load("data/panda_push/%s/%s/state_final.pt"%(pre,R),map_location="cuda")["ema"]); w.eval()
m=w.model; out={}
for B in (16,96):
    x=torch.randn(B,5,483,device="cuda"); t=torch.full((B,),500.,device="cuda")
    with torch.no_grad():
        for _ in range(10): m(x,None,t)
        torch.cuda.synchronize(); s=time.time()
        for _ in range(50): m(x,None,t)
        torch.cuda.synchronize(); ms=(time.time()-s)/50*1000
    # module timing via CUDA events
    ev=defaultdict(list); hooks=[]
    def pre_h(name):
        def f(mod,inp):
            e=torch.cuda.Event(enable_timing=True); e.record(); mod._ev0=e
        return f
    def post_h(name):
        def f(mod,inp,o):
            e=torch.cuda.Event(enable_timing=True); e.record(); ev[name].append((mod._ev0,e))
        return f
    groups={"attn":[],"mlp":[],"adaln_ln":[],"time_mlp":[m.time_mlp],"in_proj":[m.action_projection,m.particle_projection],
            "out_heads":[m.action_decoder,m.particle_decoder,m.particle_transformer.ln_f,m.particle_transformer.head]}
    for b in m.particle_transformer.blocks:
        groups["attn"].append(b.attn); groups["mlp"].append(b.mlp)
        groups["adaln_ln"]+= [b.ln_1,b.ln_2,b.gamma_1,b.beta_1,b.gamma_2,b.beta_2,b.scale_1,b.scale_2]
    for g,mods in groups.items():
        for mod in mods:
            hooks.append(mod.register_forward_pre_hook(pre_h(g))); hooks.append(mod.register_forward_hook(post_h(g)))
    with torch.no_grad():
        for _ in range(20): m(x,None,t)
    torch.cuda.synchronize()
    gt={g:sum(a.elapsed_time(b) for a,b in v)/20 for g,v in ev.items()}
    for h in hooks: h.remove()
    # kernel-level split inside attention: matmul vs softmax etc.
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10): m(x,None,t)
        torch.cuda.synchronize()
    kt=defaultdict(float)
    for e in prof.key_averages():
        kt[e.key]+=e.cuda_time_total/10/1000
    top=sorted(kt.items(),key=lambda z:-z[1])[:12]
    tot_k=sum(kt.values())
    out[B]={"ms_per_forward":ms,"module_ms":gt,"kernel_total_ms":tot_k,"top_kernels_ms":top}
    print("\nbatch %d: %.2f ms/forward (wall, synced)"%(B,ms))
    for g,v in sorted(gt.items(),key=lambda z:-z[1]): print("   %-10s %6.2f ms  %5.1f%%"%(g,v,100*v/ms))
    print("   kernels (GPU time %.2f ms):"%tot_k)
    for k,v in top: print("     %6.3f ms  %s"%(v,k[:70]))
json.dump(out,open("experiments/policy_improvement/tal_followup/profile/profile.json","w"),indent=1,default=str)

import sys, time, json, torch
sys.path.insert(0,"diffuser")
from diffuser.utils.serialization import load_config
R="3C_dlp_adalnpint_randcolor_H5_T4_seed42"; out={}
for name,pre in (("ctl10k","uwm_ctl_ext"),("qaction10k","qaction_ext")):
    mc=load_config("data/panda_push/"+pre,R,"model_config.pkl"); dc=load_config("data/panda_push/"+pre,R,"diffusion_config.pkl")
    w=dc._class(mc._class(**mc._dict),**dc._dict).cuda(); w.load_state_dict(torch.load("data/panda_push/%s/%s/state_final.pt"%(pre,R),map_location="cuda")["ema"]); w.eval()
    cond={0:torch.randn(16,480,device="cuda")*0.3,4:torch.randn(16,480,device="cuda")*0.3}
    for n in (4,10,25,50,100):
        for _ in range(2): w.conditional_sample(cond,n_steps=n)
        torch.cuda.synchronize(); t=time.time(); k=5
        for _ in range(k): w.conditional_sample(cond,n_steps=n)
        torch.cuda.synchronize(); out["%s_nfe%d"%(name,n)]=(time.time()-t)/k
        print(name,n,"%.4f s/sample"%out["%s_nfe%d"%(name,n)],flush=True)
json.dump(out,open("experiments/policy_improvement/tal_followup/nfe/timing.json","w"),indent=1)

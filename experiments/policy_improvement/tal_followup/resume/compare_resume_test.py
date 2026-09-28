import torch, sys
RUN = "3C_dlp_adalnpint_randcolor_H5_T4_seed42"
a = torch.load("data/panda_push/resumetest_A/%s/state_final.pt" % RUN, map_location="cpu")
b = torch.load("data/panda_push/resumetest_B/%s/state_final.pt" % RUN, map_location="cpu")
def cmp(x, y, name):
    bad = [k for k in x if not torch.equal(x[k], y[k])]
    mx = max(((x[k].float() - y[k].float()).abs().max().item() for k in bad), default=0.0)
    print("  %-10s %d tensors, %d differ, max|d|=%.3e" % (name, len(x), len(bad), mx)); return not bad
ok = cmp(a["model"], b["model"], "model") & cmp(a["ema"], b["ema"], "ema")
oa, ob = a["optimizer"]["state"], b["optimizer"]["state"]
ob_ok = all(torch.equal(oa[i][k], ob[i][k]) for i in oa for k in oa[i] if torch.is_tensor(oa[i][k]))
print("  optimizer  moments identical:", ob_ok)
print("  step A=%s B=%s  batches_drawn A=%s B=%s" % (a["step"], b["step"], a["batches_drawn"], b["batches_drawn"]))
ok = ok and ob_ok and a["batches_drawn"] == b["batches_drawn"]
print("EXACT RESUME BIT-IDENTICAL" if ok else "NOT IDENTICAL"); sys.exit(0 if ok else 1)

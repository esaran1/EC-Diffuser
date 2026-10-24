import numpy as np, json, itertools
from math import comb
S=['E0','E1s','E2s','E3s','E4s']
def succ(m,r,e): return np.load(f'tf_{m}_{r}_{e}_trace.npz')['d'][:,-1]<=0.04
# sanity vs json
d=json.load(open('tf_ctl10k_b96n1_R1_diagnostics.json'))
assert abs(succ('ctl10k_b96n1','R1','E0').mean()-d['control']['E0']['success'])<1e-9
def vec(m,Rs): return np.concatenate([succ(m,r,e) for r in Rs for e in S])
def mcn(a,b):
    b10=int((a&~b).sum()); b01=int((~a&b).sum()); n=b10+b01
    p=min(1,2*sum(comb(n,k) for k in range(min(b10,b01)+1))/2**n) if n else 1
    return b10,b01,p
def boot(a,b,B=20000,seed=0):
    rng=np.random.default_rng(seed); x=a.astype(float)-b.astype(float)
    i=rng.integers(0,len(x),(B,len(x))); m=x[i].mean(1); return np.percentile(m,[5,95])
R=['R1','R2']
pairs=[('qaction10k_b96n1','ctl10k_b96'),('qaction10k_b96n1','qaction10k_b96'),('ctl10k_b96n1','ctl10k_b96'),('qaction10k_b96n1','ctl10k_b96n1')]
for a,b in pairs:
    A,Bv=vec(a,R),vec(b,R); lo,hi=boot(A,Bv); b10,b01,p=mcn(A,Bv)
    print(f'{a:20s} {A.mean():.4f} vs {b:16s} {Bv.mean():.4f}  d={100*(A.mean()-Bv.mean()):+.2f}pp  90%CI[{100*lo:+.1f},{100*hi:+.1f}]  disc {b10}/{b01} McNemar p={p:.3f}  TOST3pp={"EQUIV" if lo>-0.03 and hi<0.03 else "no"}')
import sys
if len(sys.argv)>1:
    a,b,Rs=sys.argv[1],sys.argv[2],sys.argv[3].split(',')
    A,Bv=vec(a,Rs),vec(b,Rs); lo,hi=boot(A,Bv); b10,b01,p=mcn(A,Bv)
    print(f'{a} {A.mean():.4f} vs {b} {Bv.mean():.4f} d={100*(A.mean()-Bv.mean()):+.2f}pp 90%CI[{100*lo:+.1f},{100*hi:+.1f}] disc {b10}/{b01} p={p:.3f} n={len(A)}')
    for e in S: print(' ',e, round(succ(a,Rs[0],e).mean(),4), round(succ(b,Rs[0],e).mean(),4))

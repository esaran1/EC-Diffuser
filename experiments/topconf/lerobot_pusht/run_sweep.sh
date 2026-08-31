set -e
cd /home/jren313/EC-Diffuser-1/experiments/topconf/lerobot_pusht
for n in 1 2 4 5 10 20 50 100; do
  python eval_pusht.py --nsteps $n --n-episodes 50 --seed0 1000 --tag _sweep 2>&1 \
    | grep -vE "pkg_resources|UserWarning|Loading weights|true_divide|ret =" | tail -2
done

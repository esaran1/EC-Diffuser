"""Freeze one real training batch + noise + time for the real-path objective test.

The batch is drawn through the SAME dataset and the SAME seeded generator the
paired training stream uses (dataloader_seed=20260905), so the objective test is
evaluated on data training will actually see, not on synthetic tensors.

Noise (x0) and time (t) are drawn once from a separately seeded generator and
frozen, so every arm sees byte-identical inputs and any loss difference can only
come from the objective itself.

CPU only.
"""
import os
import sys

sys.argv = [
    "train.py",
    "--config", "config.pandapush_lossft_armB",
    "--num_entity", "3", "--rand_color", "--seed", "42",
    "--device", "cpu",
]

import torch
import diffuser.utils as utils
from diffuser.utils.args import ArgsParser

OUT = "experiments/policy_improvement/loss_finetune/rerun/frozen_batch.pt"
BATCH_SIZE = 32          # matches the training batch size
NOISE_SEED = 20260905    # same constant as dataloader_seed, distinct generator

args = ArgsParser().parse_args("diffusion")

dataset_config = utils.Config(
    args.loader,
    savepath=None,
    verbose=False,
    dataset_path=args.dataset_path,
    dataset_name=args.dataset,
    horizon=args.horizon,
    obs_only=args.obs_only,
    action_only=args.action_only,
    normalizer=args.normalizer,
    particle_normalizer=args.particle_normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    overfit=args.overfit,
    single_view=(args.input_type == "dlp" and not args.multiview),
)
dataset = dataset_config()

generator = torch.Generator()
generator.manual_seed(int(args.dataloader_seed))
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
    generator=generator,
)
batch = next(iter(loader))
x1, cond = utils.batchify(batch) if not isinstance(batch, (list, tuple)) else (batch[0], batch[1])

x1 = x1.float().cpu()
cond = {int(k): v.float().cpu() for k, v in cond.items()}

noise_generator = torch.Generator()
noise_generator.manual_seed(NOISE_SEED)
x0 = torch.randn(x1.shape, generator=noise_generator, dtype=torch.float32)
t = torch.rand(x1.shape[0], generator=noise_generator, dtype=torch.float32)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(
    {
        "x1": x1,
        "cond": cond,
        "x0": x0,
        "t": t,
        "batch_size": BATCH_SIZE,
        "dataloader_seed": int(args.dataloader_seed),
        "noise_seed": NOISE_SEED,
    },
    OUT,
)
print("frozen batch ->", OUT)
print("  x1", tuple(x1.shape), "cond keys", sorted(cond), "x0", tuple(x0.shape), "t", tuple(t.shape))
print("  x1 sum", float(x1.sum()), " x0 sum", float(x0.sum()), " t sum", float(t.sum()))

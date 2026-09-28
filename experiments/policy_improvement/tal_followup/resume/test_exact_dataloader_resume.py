"""Exact dataloader resume: uninterrupted vs resumed index streams. CPU only.

Uses the real torch DataLoader (1 worker, same batch size / seed / sampler
semantics as Trainer) over an index dataset of the real length, so every
generator draw -- base seed, permutation, discarded end-of-epoch permutation,
worker prefetch -- happens exactly as in training.
"""
import math
import sys
import torch

sys.path.insert(0, "diffuser")
from diffuser.utils.training import (cycle, ResumableRandomSampler,  # noqa: E402
                                     reconstruct_dataloader_generator)

N, BS, SEED = 198000, 32, 20260917
P = math.ceil(N / BS)


class Idx(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return i


def main():
    M = 400
    points = [1, 2, 4524, P - 3, P - 1, P, P + 1, 2 * P - 2, 2 * P, 2 * P + 5]
    total = max(points) + M + 1
    g0 = torch.Generator(); g0.manual_seed(SEED)
    ref_loader = cycle(torch.utils.data.DataLoader(
        Idx(), batch_size=BS, num_workers=1, shuffle=True, generator=g0))
    ref, states = [], {}
    for b in range(1, total + 1):
        ref.append(next(ref_loader).tolist())
        if b in points:
            states[b] = g0.get_state().clone()
    fails = 0
    for B in points:
        gen, into = reconstruct_dataloader_generator(
            SEED, N, BS, B, saved_state=states[B])
        loader = cycle(torch.utils.data.DataLoader(
            Idx(), batch_size=BS, num_workers=1,
            sampler=ResumableRandomSampler(Idx(), generator=gen,
                                           skip_first_samples=into * BS),
            generator=gen))
        got = [next(loader).tolist() for _ in range(M)]
        ok = got == ref[B:B + M]
        fails += not ok
        print("  resume after B=%6d (epoch %d, +%4d)  next %d batches identical: %s"
              % (B, B // P, into, M, ok))
    # negative control: a wrong batch count must be REFUSED by the state check
    try:
        reconstruct_dataloader_generator(SEED, N, BS, 4524 + P, saved_state=states[4524])
        print("  NEGATIVE CONTROL FAILED: mismatched state accepted"); fails += 1
    except RuntimeError:
        print("  negative control: mismatched batch count REFUSED (correct)")
    print("ALL PASS" if not fails else "FAILURES: %d" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

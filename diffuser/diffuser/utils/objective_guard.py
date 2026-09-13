"""Fail-loud verification that the CONSTRUCTED objective matches configured intent.

Background. A previous loss screen produced four byte-identical checkpoints
because `configuration.diffusion_wrapper_kwargs` is an explicit whitelist that
silently dropped `mask_terminal_action` and `lambda_action`. Every arm fell back
to the constructor defaults and trained the baseline objective. The pre-launch
pairing test missed it because it built the wrapper directly instead of going
through the production construction path.

The lesson encoded here: DO NOT TRUST CONFIGURATION INTENT. Verify the objective
the real training process actually constructed, and abort before optimizer step 0
if it disagrees. This module never warns-and-continues.

Two independent layers:

  1. scalar    -- the wrapper's runtime attributes equal the parsed config.
  2. structural-- the ACTUAL `loss_weight_matrix` has the intended structure,
                  and the terminal-action mask is genuinely active or inactive.

Layer 2 matters because the two knobs act in different places: `lambda_action`
scales the weight matrix, while `mask_terminal_action` acts on the conditioning
mask inside `_compute_flow_loss` and leaves the weight matrix untouched. A check
that looked only at the matrix would be blind to the mask, and a check that
looked only at scalars would be blind to a matrix built wrongly. Neither alone
is sufficient.
"""
import hashlib
import json
import os

import torch


class ObjectiveMismatch(RuntimeError):
    """Raised when the constructed objective disagrees with configured intent."""


def _hash_tensor(tensor):
    array = tensor.detach().to(torch.float64).cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def describe_objective(diffusion):
    """Return a machine-readable description of the CONSTRUCTED objective."""
    weights = diffusion.loss_weight_matrix.detach().to(torch.float64).cpu()
    action_dim = int(diffusion.action_dim)
    horizon = int(diffusion.horizon)

    action_block = weights[:, :action_dim]
    observation_block = weights[:, action_dim:]

    mask_active = bool(getattr(diffusion, "mask_terminal_action", False)) and action_dim > 0

    return {
        "mask_terminal_action": bool(getattr(diffusion, "mask_terminal_action", False)),
        "lambda_action": float(getattr(diffusion, "lambda_action", 1.0)),
        "denominator_ignores_terminal_mask": bool(
            getattr(diffusion, "denominator_ignores_terminal_mask", False)),
        "action_dim": action_dim,
        "horizon": horizon,
        "first_action_weight": float(weights[0, 0].item()) if action_dim else None,
        "loss_weight_matrix_shape": list(weights.shape),
        "loss_weight_matrix_sum": float(weights.sum().item()),
        "loss_weight_matrix_sha16": _hash_tensor(weights),
        "action_block_sum": float(action_block.sum().item()) if action_dim else 0.0,
        "observation_block_sum": float(observation_block.sum().item()),
        "action_block_row_sums": [float(v) for v in action_block.sum(dim=1)] if action_dim else [],
        # The terminal-action row of the weight matrix is NOT zeroed by masking;
        # masking happens in the conditioning mask. Recorded so a reader cannot
        # mistake a nonzero terminal weight for "masking failed".
        "terminal_action_weight_row": (
            [float(v) for v in weights[horizon - 1, :action_dim]] if action_dim else []
        ),
        "terminal_action_status": "MASKED" if mask_active else "ACTIVE",
    }


def verify_terminal_mask_effective(diffusion, cond_timesteps=(0,), batch=1):
    """Confirm the terminal-action mask is genuinely in force in the real loss path.

    Scalar attributes can be right while the mask never reaches the loss. This
    calls the wrapper's own `_make_conditioning_mask` -- the exact function
    `_compute_flow_loss` uses -- and reports whether the terminal action slice
    is actually excluded.
    """
    action_dim = int(diffusion.action_dim)
    horizon = int(diffusion.horizon)
    transition_dim = int(diffusion.transition_dim)
    probe = torch.zeros(batch, horizon, transition_dim)
    cond = {int(t): torch.zeros(batch, int(diffusion.observation_dim))
            for t in cond_timesteps}
    mask = diffusion._make_conditioning_mask(probe, cond)
    terminal_action_slice = mask[:, horizon - 1, :action_dim]

    # The loss denominator is `conditioning_mask.sum()` -- the count of ACTIVE
    # (unmasked) coordinates. That is the registered design (commit 58d07c4:
    # "GaussianDiffusion applies weights once and reduces by element count.
    # Restrict that count to unconditioned elements when masking is present"),
    # established before the terminal-action knob existed. Masking therefore
    # removes coordinates from BOTH numerator and denominator by design.
    # Recorded here so every run logs the denominator it actually trains with.
    denominator = int(mask.sum().item())

    # What the denominator would be with the terminal mask off, so the
    # renormalization factor introduced by masking is explicit in the log.
    unmasked = torch.ones_like(mask)
    for timestep in cond:
        unmasked[:, int(timestep), action_dim:] = False
    denominator_without_terminal_mask = int(unmasked.sum().item())

    # Arm B' restores the unmasked count as the divisor while keeping the masked
    # numerator, so the EFFECTIVE denominator differs from the mask's own sum.
    # Report what the loss actually divides by, not just what the mask contains.
    effective_denominator = denominator
    if (getattr(diffusion, "denominator_ignores_terminal_mask", False)
            and bool(terminal_action_slice.sum().item() == 0) and action_dim):
        effective_denominator = denominator + batch * action_dim

    ratio = (float(denominator_without_terminal_mask) / float(denominator)
             if denominator else None)

    return {
        "effective_loss_denominator": int(effective_denominator),
        "denominator_ignores_terminal_mask": bool(
            getattr(diffusion, "denominator_ignores_terminal_mask", False)),
        "terminal_action_elements_active": int(terminal_action_slice.sum().item()),
        "terminal_action_masked": bool(terminal_action_slice.sum().item() == 0),
        "conditioning_mask_total_active": denominator,
        "probe_batch": int(batch),
        "loss_denominator": denominator,
        "loss_denominator_without_terminal_mask": denominator_without_terminal_mask,
        "renormalization_ratio": ratio,
    }


def assert_objective(diffusion, expected, run_id, manifest_path=None,
                     cond_timesteps=(0,), probe_batch=1):
    """Verify the constructed objective, write a manifest, and ABORT on mismatch.

    `expected` maps knob name -> intended value, taken from the parsed config.
    Raises ObjectiveMismatch before any optimizer step if reality disagrees.
    """
    actual = describe_objective(diffusion)
    mask_probe = verify_terminal_mask_effective(diffusion, cond_timesteps, probe_batch)

    failures = []

    # --- Layer 1: scalars reached the wrapper -------------------------------
    for key in ("mask_terminal_action", "lambda_action"):
        if key not in expected:
            continue
        want, got = expected[key], actual[key]
        if isinstance(want, bool) or isinstance(got, bool):
            ok = bool(want) == bool(got)
        else:
            ok = abs(float(want) - float(got)) <= 1e-12
        if not ok:
            failures.append(
                "{}: expected {!r}, constructed {!r}".format(key, want, got))

    if "first_action_weight" in expected and actual["first_action_weight"] is not None:
        want = float(expected["first_action_weight"])
        got = float(actual["first_action_weight"])
        if abs(want - got) > 1e-9:
            failures.append(
                "first_action_weight: expected {}, constructed {}".format(want, got))

    # --- Layer 2: the mask is actually in force in the real loss path -------
    want_masked = bool(expected.get("mask_terminal_action", False))
    if mask_probe["terminal_action_masked"] != want_masked:
        failures.append(
            "terminal-action mask not in force: expected masked={}, "
            "conditioning mask reports masked={} ({} action elements still active)".format(
                want_masked,
                mask_probe["terminal_action_masked"],
                mask_probe["terminal_action_elements_active"]))

    # --- Layer 3: lambda actually scaled the ACTUAL matrix ------------------
    # This layer is the one that matters most, and it must not be circular.
    # Dividing the observed action-block sum by lambda and comparing to itself
    # would pass no matter what the matrix contained. Instead, rebuild an
    # INDEPENDENT lambda=1 reference matrix from the same wrapper and require
    # the real action block to equal it scaled by the expected lambda, while the
    # observation block is unchanged.
    #
    # This is what catches a matrix clobbered by load_state_dict: the buffer is
    # serialized into every checkpoint, so warm-starting restores the canonical
    # lambda=1 matrix over the configured one while the scalar attribute still
    # reports the configured lambda.
    want_lam = float(expected.get("lambda_action", actual["lambda_action"]))
    if actual["action_dim"]:
        saved_lambda = diffusion.lambda_action
        try:
            diffusion.lambda_action = 1.0
            unit = diffusion._make_loss_weights(
                diffusion.action_weight, diffusion.loss_discount, diffusion.loss_weights
            ).detach().to(torch.float64).cpu()
        finally:
            diffusion.lambda_action = saved_lambda

        ad = actual["action_dim"]
        unit_action_sum = float(unit[:, :ad].sum().item())
        unit_obs_sum = float(unit[:, ad:].sum().item())
        expected_action_sum = unit_action_sum * want_lam

        tol = 1e-6 * max(1.0, abs(expected_action_sum))
        if abs(actual["action_block_sum"] - expected_action_sum) > tol:
            failures.append(
                "action-block sum is {} but lambda={} requires {} "
                "(lambda=1 reference {}); the loss_weight_matrix does NOT carry "
                "the configured lambda -- was it overwritten by load_state_dict?".format(
                    actual["action_block_sum"], want_lam,
                    expected_action_sum, unit_action_sum))

        obs_tol = 1e-6 * max(1.0, abs(unit_obs_sum))
        if abs(actual["observation_block_sum"] - unit_obs_sum) > obs_tol:
            failures.append(
                "observation-block sum {} differs from the lambda-independent "
                "reference {}; lambda must not touch observation weights".format(
                    actual["observation_block_sum"], unit_obs_sum))

    manifest = {
        "run_id": run_id,
        "expected": {k: (bool(v) if isinstance(v, bool) else v) for k, v in expected.items()},
        "actual": actual,
        "terminal_mask_probe": mask_probe,
        "verdict": "FAIL" if failures else "PASS",
        "failures": failures,
    }

    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    print("[objective_guard] RUN ID: {}".format(run_id))
    print("[objective_guard]   mask_terminal_action expected={} actual={}".format(
        expected.get("mask_terminal_action", "<unset>"), actual["mask_terminal_action"]))
    print("[objective_guard]   lambda_action        expected={} actual={}".format(
        expected.get("lambda_action", "<unset>"), actual["lambda_action"]))
    print("[objective_guard]   first_action_weight  {}".format(actual["first_action_weight"]))
    print("[objective_guard]   loss_weight_matrix   shape={} sum={} sha16={}".format(
        actual["loss_weight_matrix_shape"], actual["loss_weight_matrix_sum"],
        actual["loss_weight_matrix_sha16"]))
    print("[objective_guard]   action block sum={}  observation block sum={}".format(
        actual["action_block_sum"], actual["observation_block_sum"]))
    print("[objective_guard]   terminal action      {}".format(actual["terminal_action_status"]))
    print("[objective_guard]   loss denominator     {} (unmasked would be {}, "
          "renormalization x{})".format(
              mask_probe["loss_denominator"],
              mask_probe["loss_denominator_without_terminal_mask"],
              "{:.6f}".format(mask_probe["renormalization_ratio"])
              if mask_probe["renormalization_ratio"] is not None else "n/a"))
    print("[objective_guard]   verdict: {}".format(manifest["verdict"]))

    if failures:
        raise ObjectiveMismatch(
            "constructed objective does not match configured intent for run {}:\n  - {}".format(
                run_id, "\n  - ".join(failures)))

    return manifest

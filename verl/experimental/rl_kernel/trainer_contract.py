# Copyright 2026 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contracts for the opt-in DSV4 trainer; importable without verl or GPUs."""

import hashlib
import json
from pathlib import Path

from .cluster_contract import pp_size, rollout_pp, rollout_tp, topology, worker_key

VERL_REVISION = "cc35929bf8eedf5f3eb7c4ace3441dfd9f6d4c0e"


def value(config, path, default=None):
    current = config
    for name in path.split("."):
        if not hasattr(current, "get"):
            return default
        current = current.get(name, default)
        if current is None:
            return default
    return current


def validate_trainer_config(config):
    """Fail before allocating Ray resources for unimplemented combinations."""
    topology(config)
    required = {
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": "sync",
        "trainer.resume_mode": "disable",
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "actor_rollout_ref.hybrid_engine": False,
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "actor_rollout_ref.actor.strategy": "megatron",
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.megatron.strategy": "rlk_verl_megatron",
        "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": pp_size(),
        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 1,
        "actor_rollout_ref.actor.megatron.expert_model_parallel_size": 1,
        "actor_rollout_ref.actor.megatron.context_parallel_size": 1,
        "actor_rollout_ref.actor.megatron.param_offload": False,
        "actor_rollout_ref.actor.megatron.optimizer_offload": False,
        "actor_rollout_ref.actor.megatron.use_remove_padding": False,
        "actor_rollout_ref.actor.megatron.use_fused_kernels": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": rollout_pp(),
        "actor_rollout_ref.rollout.tensor_model_parallel_size": rollout_tp(),
        "actor_rollout_ref.rollout.data_parallel_size": 1,
        "actor_rollout_ref.rollout.expert_parallel_size": 1,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.enable_sleep_mode": False,
        "actor_rollout_ref.rollout.enable_prefix_caching": False,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.logprobs_mode": "raw_logprobs",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "naive",
    }
    errors = [f"{key} must be {expected!r}" for key, expected in required.items() if value(config, key) != expected]
    for key in (
        "actor_rollout_ref.model.mtp.enable",
        "actor_rollout_ref.model.lora.rank",
        "actor_rollout_ref.model.lora_rank",
        "distillation.enable",
        "algorithm.rollout_correction.bypass_mode",
        "actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering",
        "reward.reward_model.enable",
        "actor_rollout_ref.rollout.disaggregation.enabled",
    ):
        if value(config, key, False):
            errors.append(f"{key} is unsupported in this trainer")
    if value(config, "actor_rollout_ref.rollout.n", 0) < 2:
        errors.append("GRPO requires rollout.n >= 2")
    if value(config, "actor_rollout_ref.actor.ppo_mini_batch_size") != value(config, "data.train_batch_size"):
        errors.append("ppo_mini_batch_size must equal train_batch_size (both count prompts)")
    if value(config, "trainer.v1.sync.parameter_sync_step", 1) != 1:
        errors.append("this acceptance recipe requires parameter_sync_step=1")
    sync_dir = value(config, "rlk.sync_dir")
    if not sync_dir or not Path(sync_dir).is_absolute():
        errors.append("rlk.sync_dir must be an absolute job-specific shared directory")
    if not value(config, "rlk.audit_prompts") or value(config, "rlk.audit_response_length", 0) <= 0:
        errors.append("post-update validation requires prompts and a positive response length")
    if errors:
        raise ValueError("Unsupported RLK trainer configuration: " + "; ".join(errors))


def logical_cases(input_ids):
    """Build actual token-row context, including duplicate GRPO completions.

    The fixed-size microbatch is one. Hash the whole unpadded sequence, never
    batch position, so a sample retains its canonical keys when reordered.
    """
    rows = list(input_ids.unbind())
    if not rows:
        raise ValueError("nonempty microbatch required")
    cases = {}
    for row in rows:
        tokens = row.detach().cpu().tolist()
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("nonempty unpadded integer token sequences are required")
        identity = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        # Identical GRPO completions share a logical key; matching stays unique.
        cases[identity] = dict(id=identity, prompt_token_ids=tokens, response_token_ids=[])
    return list(cases.values())


def validate_stage_acks(acks, manifest, version, previous_pids=None):
    """Every original tensor must be read back once on the two live stages."""
    if len(acks) != rollout_pp() or {ack["pipeline_stage"] for ack in acks} != set(range(rollout_pp())):
        raise RuntimeError("weight sync requires exactly two different PP stage acknowledgments")
    pids = {ack["pipeline_stage"]: worker_key(ack) for ack in acks}
    if len(set(pids.values())) != rollout_pp() or (previous_pids is not None and pids != previous_pids):
        raise RuntimeError("rollout worker PIDs changed across the continuous update")
    checked = {}
    for ack in acks:
        if ack["weight_version"] != version or not ack["parameter_load_completed"]:
            raise RuntimeError("rollout acknowledged an incomplete or stale version")
        tensors = ack["verified_original_tensors"]
        if not tensors or checked.keys() & tensors.keys():
            raise RuntimeError("empty or overlapping pipeline state acknowledgment")
        checked.update(tensors)
    if set(checked) != set(manifest) or any(
        any(checked[name].get(key) != item[key] for key in ("sha256", "dtype", "shape"))
        for name, item in manifest.items()
    ):
        raise RuntimeError("rollout readback does not cover the complete actor state")
    return pids


def strict_response_parity(actor, rollout, response_mask):
    """Compare finite FP32 response scores as bits; never round or use allclose."""
    import torch

    if actor.dtype != torch.float32 or rollout.dtype != torch.float32:
        raise TypeError("strict parity requires original FP32 actor and rollout scores")
    if actor.shape != rollout.shape or actor.shape != response_mask.shape:
        raise ValueError("response score shapes/mask must match")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response mask must contain only zero or one")
    selected = response_mask.bool()
    left, right = actor[selected].contiguous(), rollout[selected].contiguous()
    if not left.numel() or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("response parity requires nonempty finite observations")
    mismatch = int((left.view(torch.int32) != right.view(torch.int32)).sum())
    if mismatch:
        index = int(torch.nonzero(left.view(torch.int32) != right.view(torch.int32))[0])
        position = torch.nonzero(selected)[index].tolist()
        raise RuntimeError(
            f"actor/rollout raw logprob mismatch: {mismatch}/{left.numel()}; "
            f"first_position={position}, actor={float(left[index])}, rollout={float(right[index])}, "
            f"actor_bits=0x{int(left.view(torch.int32)[index]) & 0xFFFFFFFF:08x}, "
            f"rollout_bits=0x{int(right.view(torch.int32)[index]) & 0xFFFFFFFF:08x}"
        )
    delta = (left.double() - right.double()).abs()
    return dict(
        expected_tokens=int(selected.sum()),
        compared_tokens=left.numel(),
        mismatch_count=mismatch,
        finite=True,
        mean_abs_diff=float(delta.mean()),
        max_abs_diff=float(delta.max()),
    )

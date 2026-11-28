# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- install opt-in ABI adapters before framework imports.
"""Actual GPU startup diagnostics; never count these as GRPO acceptance.

Run actor with torchrun and rollout as one driver. Target topology is taken from
an immutable profile, and every failure keeps its original traceback.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from rl_engine.integrations.orchestrators.verl.native_imports import install_native_import_adapters

install_native_import_adapters()

import torch


def actor(profile, model_path, output):
    import torch.distributed as dist
    from rl_engine.integrations.common.device_identity import hip_device_identity
    from rl_engine.integrations.orchestrators.verl.native_engine import RLKNativeMcoreEngineConfig

    from verl.trainer.config import CheckpointConfig
    from verl.utils.profiler.config import ProfilerConfig
    from verl.workers.config import HFModelConfig, McoreOptimizerConfig, TrainingWorkerConfig
    from verl.workers.engine_workers import TrainingWorker

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    target = profile["actor"]
    if dist.get_world_size() != target["gpus"]:
        raise ValueError("torchrun world size differs from target actor topology")
    canary = torch.arange(16, device="cuda", dtype=torch.float32).sum()
    torch.cuda.synchronize()
    identity = dict(
        rank=dist.get_rank(),
        local_rank=local_rank,
        device=hip_device_identity(),
        canary=float(canary),
        requested=target,
        gpu_initialized=True,
    )
    (output / f"actor-rank-{dist.get_rank()}-gpu.json").write_text(json.dumps(identity, indent=2) + "\n")
    cfg = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(path=str(model_path), use_remove_padding=False),
        engine_config=RLKNativeMcoreEngineConfig(
            rlk_mhc_backend="rocm_triton",
            rlk_allow_right_padding=False,
            tensor_model_parallel_size=target["tp"],
            pipeline_model_parallel_size=target["pp"],
            expert_model_parallel_size=target["ep"],
            expert_tensor_parallel_size=target["etp"],
            context_parallel_size=1,
            sequence_parallel=target["sequence_parallel"],
            param_offload=False,
            optimizer_offload=False,
            forward_only=True,
            use_distributed_optimizer=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            infer_micro_batch_size_per_gpu=1,
            override_transformer_config=dict(
                gradient_accumulation_fusion=False,
                recompute_granularity=None,
                apply_rope_fusion=False,
                moe_permute_fusion=False,
                moe_shared_expert_overlap=False,
            ),
        ),
        optimizer_config=McoreOptimizerConfig(lr=1e-4, total_training_steps=3, weight_decay=0.0, clip_grad=1.0),
        checkpoint_config=CheckpointConfig(),
        profiler_config=ProfilerConfig(tool=None, save_path=str(output), tool_config={}),
    )
    worker = TrainingWorker(cfg)
    worker.reset()
    engine = worker.engine
    shard = engine._rlk_models[0]
    from examples.operators.dsv4_4l.actor_service import batch

    cases = [dict(id="startup-short", prompt_token_ids=[10, 11, 12, 13, 14], response_token_ids=[15, 16, 17])]
    engine.rlk_logical_cases = cases
    result = worker.infer_batch(batch(cases))
    if dist.get_rank() == dist.get_world_size() - 1:
        raw = result["raw_log_probs"].values().detach().cpu()
        if raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
            raise ValueError("independent actor forward did not return finite original FP32 scores")
        torch.save(
            dict(cases=cases, raw_log_probs=raw, scope="startup only; no cross-topology comparison"),
            output / "actor-startup-scores.pt",
        )
    state = dict(
        identity,
        status="STARTUP_ONLY",
        original_layers=list(shard._rlk_checkpoint_stage.layer_ids),
        optimizer_updates=0,
        trainable=[
            dict(name=n, shape=list(p.shape), dtype=str(p.dtype))
            for n, p in shard.named_parameters()
            if p.requires_grad
        ],
    )
    (output / f"actor-rank-{dist.get_rank()}-loaded.json").write_text(json.dumps(state, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


def rollout(profile, model_path, output):
    from vllm import LLM, SamplingParams

    from verl.experimental.rl_kernel.runtime import prepare_worker

    prepare_worker()
    target = profile["rollout"]
    os.environ["RLK_VERL_MHC_BACKEND"] = "rocm_triton"
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=target["tp"],
        pipeline_parallel_size=target["pp"],
        distributed_executor_backend="mp",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_sleep_mode=True,
        gpu_memory_utilization=0.30,
        max_model_len=2304,
        max_num_seqs=2,
        max_num_batched_tokens=2304,
        logprobs_mode="raw_logprobs",
        worker_cls="verl.experimental.rl_kernel.rollout_worker.RLKVerlDSV4Worker",
    )
    generated = llm.generate(
        [dict(prompt_token_ids=[10, 11, 12, 13, 14])],
        SamplingParams(temperature=1, top_k=-1, top_p=1, max_tokens=3, logprobs=1),
    )
    receipt = dict(
        status="STARTUP_ONLY",
        requested=target,
        optimizer_updates=0,
        generations=[
            dict(prompt_token_ids=r.prompt_token_ids, response_ids=list(r.outputs[0].token_ids)) for r in generated
        ],
    )
    (output / "rollout-generated.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Verify pinned quantized sleep capability directly. No state continuity claim is inferred from success.
    began = time.monotonic()
    llm.sleep(level=1)
    receipt["sleep_seconds"] = time.monotonic() - began
    began = time.monotonic()
    llm.wake_up()
    receipt["wake_seconds"] = time.monotonic() - began
    (output / "rollout-sleep-wake.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--role", choices=["actor", "rollout"], required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        globals()[args.role](json.loads(args.profile.read_text()), args.model, args.output)
    except BaseException as e:
        rank = os.environ.get("RANK", "driver")
        (args.output / f"{args.role}-{rank}-error.json").write_text(
            json.dumps(
                dict(
                    status="FAIL",
                    role=args.role,
                    rank=rank,
                    error_type=type(e).__name__,
                    error=str(e),
                    traceback=traceback.format_exc(),
                    optimizer_updates=0,
                ),
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()

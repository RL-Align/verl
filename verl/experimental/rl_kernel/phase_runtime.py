# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- install the explicit opt-in framework ABI adapters first.
"""Real seven-layer actor-TP1 scoring and native verl rollout phase processes.

Each invocation owns one phase. Process exit releases that phase's GPU contexts;
no worker residency or training acceptance is inferred from these entrypoints.
"""

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import struct
import time
import traceback
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from rl_engine.integrations.orchestrators.verl.native_imports import install_native_import_adapters

install_native_import_adapters()

import torch
from verl.experimental.rl_kernel.dp_shard import gather_in_order, require_even, shard_cases, shard_indices


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def raw_bytes(values):
    if isinstance(values, torch.Tensor):
        if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()):
            raise ValueError("scores must be original finite FP32")
        return values.detach().cpu().contiguous().numpy().tobytes()
    # The native token API serializes original FP32 values as Python floats.
    # Require exact representability; never round an arbitrary FP64 score.
    out = bytearray()
    for value in values:
        if not math.isfinite(value):
            raise ValueError("native score contains NaN/Inf")
        packed = struct.pack("<f", value)
        if struct.unpack("<f", packed)[0] != value:
            raise ValueError("native API score lost its original FP32 representation")
        out.extend(packed)
    return bytes(out)


def suite(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not value.get("samples"):
        raise ValueError("fixed suite cannot be empty")
    ids = set()
    for case in value["samples"]:
        if case["id"] in ids or not case["prompt_ids"] or not case["response_ids"]:
            raise ValueError("duplicate or empty logical input")
        ids.add(case["id"])
        if case["position_ids"] != list(range(len(case["prompt_ids"]) + len(case["response_ids"]))):
            raise ValueError("this native phase requires the actual contiguous original positions")
        if case["mask"] != "full-causal-unpadded-all-response-tokens":
            raise ValueError("logical mask differs from native full causal scoring")
    return value["samples"], hashlib.sha256(raw).hexdigest()


def groups(cases):
    ordered = defaultdict(list)
    for case in cases:
        ordered[case.get("batch_id", case["id"])].append(case)
    return ordered.items()


def save_scores(root, cases, scores, producer, suite_hash, snapshot_hash):
    if len(cases) != len(scores):
        raise ValueError("missing response score row")
    rows = []
    for index, (case, values) in enumerate(zip(cases, scores, strict=True)):
        data = raw_bytes(values)
        if len(data) != len(case["response_ids"]) * 4:
            raise ValueError("missing response-token scores")
        name = f"sample-{index:04d}.f32"
        (root / name).write_bytes(data)
        rows.append(
            dict(
                id=case["id"],
                response_ids=case["response_ids"],
                expected_tokens=len(case["response_ids"]),
                file=name,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    manifest = dict(
        dtype="float32",
        encoding="little-endian-ieee754",
        input_suite_sha256=suite_hash,
        snapshot_sha256=snapshot_hash,
        temperature=1,
        top_k=-1,
        top_p=1,
        producer=producer,
        rows=rows,
    )
    write(root / "scores.json", manifest)
    return manifest


def make_data(cases, *, microbatch=1):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    tokens = [torch.tensor(c["prompt_ids"] + c["response_ids"], dtype=torch.int64) for c in cases]
    masks = [torch.tensor([0] * len(c["prompt_ids"]) + [1] * len(c["response_ids"])) for c in cases]
    data = TensorDict(
        dict(
            input_ids=torch.nested.as_nested_tensor(tokens, layout=torch.jagged),
            loss_mask=torch.nested.as_nested_tensor(masks, layout=torch.jagged),
            temperature=torch.ones(len(cases)),
        ),
        batch_size=[len(cases)],
    )
    tu.assign_non_tensor(
        data,
        global_token_num=[len(t) for t in tokens],
        compute_loss=False,
        disable_auto_offload=True,
        micro_batch_size_per_gpu=microbatch,
        calculate_entropy=False,
    )
    return data


def _score_rows(result, cases):
    """Response-token FP32 scores per case on an output-owning rank, else None."""
    if result is None:
        return None
    return [
        row[len(case["prompt_ids"]) - 1 : len(case["prompt_ids"]) + len(case["response_ids"]) - 1]
        .detach()
        .cpu()
        .clone()
        for row, case in zip(result["log_probs"].unbind(), cases, strict=True)
    ]


def dp_coordinates(worker):
    """(dp_rank, dp_size, owns_outputs) of this rank in the actor engine."""
    engine = worker.engine
    return engine.get_data_parallel_rank(), engine.get_data_parallel_size(), engine.is_mp_src_rank_with_outputs()


def is_score_writer(worker):
    """Single writer of score files: the output-owning rank of DP replica 0.

    With PP2/DP1 this is the last pipeline rank, the only rank that wrote scores before DP sharding.
    """
    dp_rank, _, owner = dp_coordinates(worker)
    return owner and dp_rank == 0


def score_batch(worker, batch_cases):
    """Score one logical batch across DP replicas; every rank returns all scores in case order.

    Each DP rank scores its padded shard as one microbatch, so all DP ranks run the
    same number of forward calls (required when the MoE all-to-all spans DP). With
    DP1 this is exactly ``infer_batch(make_data(batch_cases, microbatch=len(batch_cases)))``.
    """
    dp_rank, dp_size, owner = dp_coordinates(worker)
    local, _ = shard_cases(batch_cases, dp_rank, dp_size)
    rows = _score_rows(worker.infer_batch(make_data(local, microbatch=len(local))), local)
    return gather_in_order(rows, len(batch_cases), dp_rank, dp_size, owner)


def score_single_rounds(worker, cases):
    """Score each case alone (microbatch 1) in DP-sharded rounds; all ranks get all scores in order.

    Every round each DP rank scores one case, or a padding duplicate whose output is dropped.
    """
    dp_rank, dp_size, owner = dp_coordinates(worker)
    local, _ = shard_cases(cases, dp_rank, dp_size)
    rows = []
    for case in local:
        scored = _score_rows(worker.infer_batch(make_data([case])), [case])
        rows.append(scored[0] if scored is not None else None)
    return gather_in_order(rows if owner else [], len(cases), dp_rank, dp_size, owner)


def build_actor(args, *, training=False):
    import torch.distributed as dist
    from rl_engine.integrations.common.device_identity import hip_device_identity

    from verl.experimental.rl_kernel.trainer_engine import RLKVerlMcoreEngineConfig
    from verl.trainer.config import CheckpointConfig
    from verl.utils.profiler.config import ProfilerConfig
    from verl.workers.config import HFModelConfig, McoreOptimizerConfig, TrainingWorkerConfig
    from verl.workers.engine_workers import TrainingWorker

    profile = json.loads(args.profile.read_text())
    if profile["id"] not in ("N1", "N2", "N3", "N4", "N5"):
        raise ValueError("only the revised primary N1-N5 actor profiles may execute here")
    target = profile["actor"]
    if target["tp"] != 1 or target["etp"] != 1 or target["cp"] != 1:
        raise ValueError("native DSV4 actor TP1/ETP1/CP1 restrictions are retained")
    if target["pp"] != 2 or target["gpus"] != target["pp"] * target["dp"]:
        raise ValueError("native DSV4 actor requires PP2 with world = PP x dense DP")
    if target["ep"] not in (1, 4):
        raise ValueError("native DSV4 actor supports EP1 or EP4 only")
    if target["ep"] == 4 and (target["dp"] != 4 or target["expert_dp"] != 1):
        raise ValueError("EP4 requires dense DP4 and expert DP1 (EP group = the DP ranks of a PP stage)")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    if dist.get_world_size() != target["gpus"]:
        raise ValueError("actual actor world differs from target")
    cfg = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(path=str(args.model), use_remove_padding=False),
        engine_config=RLKVerlMcoreEngineConfig(
            rlk_weight_version=args.version,
            rlk_mhc_backend="rocm_triton",
            rlk_allow_right_padding=True,
            rlk_evidence_dir=str(args.output),
            rlk_require_update_forward_parity=training,
            # Sparse attention backward needs ~13.2 GB at 8,192 queries (1024+7168).
            rlk_sparse_workspace_bytes=16 << 30,
            # One canonical parameter-gradient reduction per microbatch, accumulated by DDP in
            # fixed microbatch order: a step-wide reduction keeps every saved row of a
            # 128 x 8k-token batch on the device (hundreds of GiB).
            rlk_canonical_scope="microbatch",
            tensor_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=target["ep"],
            context_parallel_size=1,
            sequence_parallel=False,
            param_offload=False,
            optimizer_offload=False,
            forward_only=not training,
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
        checkpoint_config=CheckpointConfig(save_contents=["optimizer", "extra"], load_contents=["optimizer", "extra"]),
        profiler_config=ProfilerConfig(tool=None, save_path=str(args.output), tool_config={}),
    )
    worker = TrainingWorker(cfg)
    worker.reset()
    from megatron.core import parallel_state as mpu

    identity = dict(
        engine="verl.TrainingWorker.infer_batch",
        weight_version=args.version,
        role="actor",
        rank=dist.get_rank(),
        dp_rank=worker.engine.get_data_parallel_rank(),
        dp_size=worker.engine.get_data_parallel_size(),
        ep_rank=mpu.get_expert_model_parallel_rank(),
        ep_size=mpu.get_expert_model_parallel_world_size(),
        pid=os.getpid(),
        topology=target,
        original_layers=list(worker.engine._rlk_models[0]._rlk_checkpoint_stage.layer_ids),
        **hip_device_identity(),
    )
    write(args.output / f"actor-rank-{dist.get_rank()}.json", identity)
    return worker, identity


def actor_score(args):
    import torch.distributed as dist
    from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import digest

    cases, suite_hash = suite(args.suite)
    worker, identity = build_actor(args)
    if args.version:
        raise ValueError("noninitial scoring must run through restored training phase, not original weights")
    scores, batches = [], []
    for batch_id, batch_cases in groups(cases):
        start = time.monotonic()
        import re
        from contextlib import ExitStack

        with ExitStack() as trace:
            pattern = os.environ.get("RLK_TRACE_AUDIT_PATTERN")
            if pattern and re.fullmatch(pattern, batch_id):
                from examples.operators.dsv4_4l.boundary_trace import capture

                for local, model in enumerate(worker.engine._rlk_models):
                    trace.callback(
                        capture(
                            model,
                            args.output / "traces" / f"actor-rank-{dist.get_rank()}-chunk-{local}-{batch_id}",
                            "actor",
                        )
                    )
            # Every rank receives all scores of the batch in original case order.
            scores.extend(score_batch(worker, batch_cases))
        batches.append(
            dict(batch_id=batch_id, samples=[c["id"] for c in batch_cases], seconds=time.monotonic() - start)
        )
    # Native PP export gathers original names/state; quantized base remains in the pinned source model.
    tensors, _ = worker.engine.get_per_tensor_param(base_sync_done=True)
    weights = {name: tensor.detach().cpu().clone() for name, tensor in tensors}
    manifest = {name: dict(shape=list(t.shape), dtype=str(t.dtype), sha256=digest(t)) for name, t in weights.items()}
    if dist.get_rank() == 0:
        torch.save(dict(weight_version=0, weights=weights, manifest=manifest), args.output / "policy-v0.pt")
        write(args.output / "policy-v0-manifest.json", manifest)
    if is_score_writer(worker):
        save_scores(args.output, cases, scores, identity, suite_hash, args.snapshot_sha256)
        write(args.output / "actor-batches.json", batches)
    dist.barrier()
    dist.destroy_process_group()


def export_policy(worker, destination):
    import torch.distributed as dist
    from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import digest

    parameters, _ = worker.engine.get_per_tensor_param(base_sync_done=True)
    weights = {name: tensor.detach().cpu().clone() for name, tensor in parameters}
    manifest = {name: dict(shape=list(t.shape), dtype=str(t.dtype), sha256=digest(t)) for name, t in weights.items()}
    if dist.get_rank() == 0:
        torch.save(
            dict(weight_version=worker.engine.rlk_weight_version, weights=weights, manifest=manifest), destination
        )
    dist.barrier()
    return manifest


def actor_update(args):
    import functools

    import numpy as np
    import torch.distributed as dist
    from rl_engine.integrations.common.response_bitwise import compare_fp32_bytes

    from verl.experimental.rl_kernel.phase_checkpoint import load_rank, save_rank
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import ActorConfig
    from verl.workers.utils.losses import ppo_loss

    if not all((args.batch, args.provenance, args.next_checkpoint)):
        raise ValueError("training requires original generated batch, source provenance and complete next checkpoint")
    generated = json.loads(args.batch.read_text())
    if generated["weight_version"] != args.version or generated["generation_source"] != "verl.vLLMHttpServer.generate":
        raise ValueError("training batch came from a stale or different generation path")
    cases = generated["samples"]
    if not cases or any(not c["response_ids"] for c in cases):
        raise ValueError("all generated responses must be nonempty")
    provenance = json.loads(args.provenance.read_text())
    provenance = dict(
        provenance,
        group=json.loads(args.profile.read_text())["id"],
        actor=json.loads(args.profile.read_text())["actor"],
    )
    worker, identity = build_actor(args, training=True)
    if args.version:
        if not args.resume:
            raise ValueError("cannot reset optimizer history when rebuilding a later actor phase")
        receipt = load_rank(worker.engine, args.resume, provenance)
        write(args.output / f"restore-rank-{dist.get_rank()}.json", receipt)
    elif args.resume:
        raise ValueError("initial actor must not load an unrelated update history")
    # Always recompute independently on the actual native actor before PPO.
    began = time.monotonic()
    dp_rank, dp_size, _ = dp_coordinates(worker)
    # Training never pads (a duplicate would add loss); GRPO batches are 4 or 128.
    require_even(len(cases), dp_size)
    # DP-sharded lockstep rounds; every rank ends with all actor scores in case order.
    actor_scores = score_single_rounds(worker, cases)
    if len(actor_scores) != len(cases):
        raise RuntimeError("actor scoring did not cover every training sample")
    comparisons = []
    for index, (case, actor) in enumerate(zip(cases, actor_scores, strict=True)):
        left, right = raw_bytes(actor), raw_bytes(case["rollout_logprobs"])
        if dist.get_rank() == 0:
            (args.output / f"train-{index:04d}-actor.f32").write_bytes(left)
            (args.output / f"train-{index:04d}-rollout.f32").write_bytes(right)
        comparisons.append(compare_fp32_bytes(left, right, len(case["response_ids"])))
    parity = dict(
        expected_tokens=sum(r["expected_tokens"] for r in comparisons),
        compared_tokens=sum(r["compared_tokens"] for r in comparisons),
        mismatch_count=sum(r["mismatch_count"] for r in comparisons),
        rows=comparisons,
        actor_logprob_seconds=time.monotonic() - began,
    )
    write(args.output / f"training-parity-rank-{dist.get_rank()}.json", parity)
    if parity["mismatch_count"]:
        raise RuntimeError("original FP32 training-batch parity failed before backward/update")
    lengths = [len(c["response_ids"]) for c in cases]
    response_mask = torch.zeros(len(cases), max(lengths), dtype=torch.float32)
    rewards = torch.zeros_like(response_mask)
    for i, case in enumerate(cases):
        response_mask[i, : lengths[i]] = 1
        rewards[i, lengths[i] - 1] = case["raw_reward"]
    if not bool(torch.isfinite(rewards).all()):
        raise RuntimeError("native reward contains NaN/Inf")
    advantages, _ = compute_grpo_outcome_advantage(
        rewards, response_mask, np.array([c["group_id"] for c in cases]), norm_adv_by_std_in_grpo=True
    )
    sample_advantages = torch.stack([advantages[i, lengths[i] - 1] for i in range(len(cases))])  # one scalar per sample
    write(
        args.output / f"reward-advantage-rank-{dist.get_rank()}.json",
        dict(
            raw_rewards=[c["raw_reward"] for c in cases],
            group_ids=[c["group_id"] for c in cases],
            nonzero_advantages=int(torch.count_nonzero(advantages)),
            advantage_samples=len(cases),
            advantages=[float(a) for a in sample_advantages],
            advantage_mean_abs=float(sample_advantages.abs().mean()),
            advantage_std=float(sample_advantages.std()) if len(cases) > 1 else 0.0,
            advantage_nonzero_samples=int(torch.count_nonzero(sample_advantages)),
            finite=bool(torch.isfinite(advantages).all()),
            advantage_source="verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage",
        ),
    )
    if not bool(torch.isfinite(advantages).all()) or not bool(torch.count_nonzero(advantages)):
        raise RuntimeError("no effective GRPO update: native advantages are zero or nonfinite")
    # Advantages above are global and identical on every rank; train only on this DP rank's shard.
    mine = shard_indices(len(cases), dp_rank, dp_size)
    local_cases = [cases[i] for i in mine]
    data = make_data(local_cases)
    nested = lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    data["prompts"] = nested([torch.tensor(c["prompt_ids"], dtype=torch.int64) for c in local_cases])
    data["responses"] = nested([torch.tensor(c["response_ids"], dtype=torch.int64) for c in local_cases])
    data["response_mask"] = nested([torch.ones(lengths[i]) for i in mine])
    data["old_log_probs"] = nested([actor_scores[i] for i in mine])
    data["advantages"] = nested([advantages[i, : lengths[i]] for i in mine])
    # batch_num_tokens is the GLOBAL response-token count. forward_backward_batch recomputes the same
    # value by all-reducing the local loss_mask sum over the DP group, so this agrees with it.
    tu.assign_non_tensor(
        data,
        batch_num_tokens=sum(lengths),
        dp_size=dp_size,
        global_batch_size=len(cases),
        update_lr_scheduler=True,
        compute_loss=True,
    )
    loss_config = ActorConfig(
        strategy="megatron",
        rollout_n=generated["request"]["samples_per_prompt"],
        ppo_mini_batch_size=len(cases),
        ppo_micro_batch_size_per_gpu=1,
        use_dynamic_bsz=False,
        use_kl_loss=False,
        entropy_coeff=0.0,
        loss_agg_mode="token-mean",
    )
    worker.set_loss_fn(functools.partial(ppo_loss, config=loss_config))
    began = time.monotonic()
    result = worker.train_batch(data)
    native_train_seconds = time.monotonic() - began
    if worker.engine.rlk_weight_version != args.version + 1:
        raise RuntimeError("native optimizer did not advance exactly one weight version")
    # Single writer (DP replica 0's output owner); other DP ranks would race on the same path.
    if result is not None and is_score_writer(worker):
        torch.save(result, args.output / "native-train-output.pt")
    checkpoint = save_rank(worker.engine, args.next_checkpoint, provenance)
    manifest = export_policy(worker, args.output / f"policy-v{args.version + 1}.pt")
    write(
        args.output / f"updated-rank-{dist.get_rank()}.json",
        dict(
            identity,
            weight_version=worker.engine.rlk_weight_version,
            actor_train_seconds=native_train_seconds,
            checkpoint_and_export_seconds=time.monotonic() - began - native_train_seconds,
            actor_train_and_checkpoint_seconds=time.monotonic() - began,
            checkpoint=checkpoint,
            exported_tensors=len(manifest),
            native_loss="verl.workers.utils.losses.ppo_loss",
            scope="actual update receipt; post-update rollout and independent verification are still required",
        ),
    )
    dist.destroy_process_group()


def actor_audit(args):
    import torch.distributed as dist

    from verl.experimental.rl_kernel.phase_checkpoint import load_rank, state_digest

    if not args.resume or not args.provenance:
        raise ValueError("post-update audit must restore complete actor state")
    provenance = dict(
        json.loads(args.provenance.read_text()),
        group=json.loads(args.profile.read_text())["id"],
        actor=json.loads(args.profile.read_text())["actor"],
    )
    worker, identity = build_actor(args, training=True)
    receipt = load_rank(worker.engine, args.resume, provenance)
    write(args.output / f"restore-rank-{dist.get_rank()}.json", receipt)
    before = state_digest(worker.engine.optimizer.state_dict())
    cases, suite_hash = suite(args.suite)
    if args.batch:
        generated = json.loads(args.batch.read_text())
        if generated["weight_version"] != args.version:
            raise ValueError("fresh audit batch is stale")
        cases = generated["samples"]
        suite_path = args.batch.parent / "generated-suite.json"
        expected_cases, suite_hash = suite(suite_path)
        if [c["id"] for c in cases] != [c["id"] for c in expected_cases]:
            raise ValueError("fresh generation and logical suite differ")
    scores = []
    for batch_id, batch_cases in groups(cases):
        import re
        from contextlib import ExitStack

        # Optional diagnostic capture; unset env var means no behaviour change.
        with ExitStack() as trace:
            pattern = os.environ.get("RLK_TRACE_AUDIT_PATTERN")
            if pattern and re.fullmatch(pattern, batch_id):
                from examples.operators.dsv4_4l.boundary_trace import capture

                for local, model in enumerate(worker.engine._rlk_models):
                    trace.callback(
                        capture(
                            model,
                            args.output / "traces" / f"actor-rank-{dist.get_rank()}-chunk-{local}-{batch_id}",
                            "actor",
                        )
                    )
            scores.extend(score_batch(worker, batch_cases))
    if is_score_writer(worker):
        save_scores(args.output, cases, scores, identity, suite_hash, args.snapshot_sha256)
    if state_digest(worker.engine.optimizer.state_dict()) != before:
        raise RuntimeError("audit changed optimizer history")
    dist.barrier()
    dist.destroy_process_group()


async def rollout_phase(args):
    import ray

    from verl.experimental.rl_kernel.trainer_rollout import RLKTrainerVLLMServer
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class PhaseServer(vLLMHttpServer):
        rlk_score_sequences = RLKTrainerVLLMServer.rlk_score_sequences

        def _preprocess_engine_kwargs(self, engine_kwargs):
            super()._preprocess_engine_kwargs(engine_kwargs)
            engine_kwargs["worker_cls"] = "verl.experimental.rl_kernel.rollout_worker.RLKVerlDSV4Worker"
            engine_kwargs["distributed_executor_backend"] = "mp"
            engine_kwargs["pipeline_parallel_size"] = 1

    cases, suite_hash = suite(args.suite)
    request = json.loads(args.batch.read_text()) if args.action == "rollout-generate" and args.batch else None
    target = json.loads(args.profile.read_text())["rollout"]
    if target["pp"] != 1 or target["ep"] != 1 or target["instances"] != 1:
        raise ValueError("revised rollout requires one native PP1/EP1 instance")
    if not args.policy:
        raise ValueError("rollout must receive and read back the actual actor policy")
    os.environ["RLK_PHASE_BOOTSTRAP_POLICY"] = str(args.policy)
    os.environ.pop("VLLM_PP_LAYER_PARTITION", None)
    os.environ["RLK_VERL_MHC_BACKEND"] = "rocm_triton"
    os.environ["RLK_VERL_SPARSE_WORKSPACE_BYTES"] = str(4 << 30)
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        namespace="dsv4-phase-" + uuid4().hex,
        _temp_dir="/tmp/rv4/" + uuid4().hex[:8],
    )
    config = RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=target["tp"],
        pipeline_model_parallel_size=1,
        expert_parallel_size=1,
        data_parallel_size=1,
        enforce_eager=True,
        enable_sleep_mode=False,
        enable_prefix_caching=False,
        calculate_log_probs=True,
        logprobs_mode="raw_logprobs",
        temperature=1,
        top_p=1,
        top_k=-1,
        max_model_len=8193,
        max_num_seqs=128,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.30,
        load_format="safetensors",
        engine_kwargs=dict(vllm=dict(compilation_config=dict(cudagraph_mode="NONE"))),
    )
    server = PhaseServer(
        config=config,
        model_config=HFModelConfig(path=str(args.model), use_remove_padding=False),
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=target["gpus"],
        nnodes=1,
        cuda_visible_devices=os.environ["HIP_VISIBLE_DEVICES"],
    )
    try:
        await server.launch_server()
        server.global_steps = args.version
        placement = await server.collective_rpc("rlk_placement")
        write(args.output / "rollout-ranks.json", placement)
        acknowledgments = await server.collective_rpc("rlk_phase_bootstrap_ack")
        if len(acknowledgments) != target["gpus"] or {row["tensor_parallel_rank"] for row in acknowledgments} != set(
            range(target["tp"])
        ):
            raise RuntimeError("actual rollout TP rank coverage differs")
        if any(row["weight_version"] != args.version for row in acknowledgments):
            raise RuntimeError("rollout bootstrap acknowledged a stale policy")
        write(args.output / "policy-readback.json", acknowledgments)
        raw_readbacks = await server.collective_rpc(
            "rlk_phase_readback_dump", args=(str(args.policy), str(args.output / "readback"))
        )
        write(args.output / "policy-raw-readback.json", raw_readbacks)
        if args.action == "rollout-generate":
            if request is None:
                raise ValueError("native generation needs a locked prompt/reward request")
            from examples.operators.dsv4_4l.copy_task import compute_score as copy_score
            from examples.operators.dsv4_4l.long_context_copy import compute_score as lcc_score

            from verl.utils.reward_score import default_compute_score

            cap, count = request["max_tokens"], request["samples_per_prompt"]
            if count <= 1 or cap <= 0 or cap > 7168:
                raise ValueError("invalid GRPO response count or cap")
            expanded = [(p, i) for p in request["prompts"] for i in range(count)]
            if any(not p["prompt_ids"] or len(p["prompt_ids"]) > 1024 for p, _ in expanded):
                raise ValueError("original prompt exceeds the requested 1024-token cap")
            began = time.monotonic()
            outputs = await asyncio.gather(
                *[
                    server.generate(
                        prompt_ids=p["prompt_ids"],
                        request_id=f"phase-{args.version}-{j}-{uuid4().hex}",
                        sampling_params=dict(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=cap, logprobs=True),
                    )
                    for j, (p, _) in enumerate(expanded)
                ]
            )
            elapsed = time.monotonic() - began
            generated = []
            for j, ((prompt, replicate), output) in enumerate(zip(expanded, outputs, strict=True)):
                response = list(output.token_ids)
                if not response or len(response) > cap:
                    raise RuntimeError("empty or over-limit response; do not skip failed samples")
                values = output.log_probs.tolist() if isinstance(output.log_probs, torch.Tensor) else output.log_probs
                raw_bytes(values)  # Check exact original FP32 serialization, with no rounding.
                if len(values) != len(response):
                    raise RuntimeError("native sampled logprob count differs from response")
                text = server.model_config.tokenizer.decode(response, skip_special_tokens=True)
                if prompt["data_source"] == "rlk/exact-copy":
                    scoring = copy_score
                elif prompt["data_source"] == "rlk/long-context-copy":
                    scoring = lcc_score
                else:
                    scoring = default_compute_score
                source = prompt["data_source"]
                if source.startswith("BytedTsinghua-SIA/"):
                    source = "math_dapo"
                extra_info = prompt.get("extra_info", {})
                if source == "rlk/long-context-copy":
                    # Token-level reward: score the actual generated ids (target_ids/eos_token_id are in the request).
                    extra_info = dict(extra_info, response_ids=response, max_tokens=cap)
                reward = scoring(
                    data_source=source,
                    solution_str=text,
                    ground_truth=prompt["ground_truth"],
                    extra_info=extra_info,
                )
                score = reward["score"] if isinstance(reward, dict) else reward
                generated.append(
                    dict(
                        id=f"fresh-{args.version}-{j}",
                        batch_id=f"fresh-single-{j}",
                        group_id=prompt["group_id"],
                        prompt_ids=prompt["prompt_ids"],
                        response_ids=response,
                        rollout_logprobs=values,
                        raw_reward=float(score),
                        reward_evidence=reward,
                        reward_source=scoring.__module__ + "." + scoring.__name__,
                        reward_source_file=inspect.getfile(scoring),
                        reward_source_sha256=hashlib.sha256(Path(inspect.getfile(scoring)).read_bytes()).hexdigest(),
                        position_ids=list(range(len(prompt["prompt_ids"]) + len(response))),
                        mask="full-causal-unpadded-all-response-tokens",
                    )
                )
            generated_suite = dict(
                samples=generated, scope="actual native verl generated samples at this policy version"
            )
            suite_path = args.output / "generated-suite.json"
            write(suite_path, generated_suite)
            _, generated_hash = suite(suite_path)
            save_scores(
                args.output,
                generated,
                [c["rollout_logprobs"] for c in generated],
                dict(
                    engine="verl.vLLMHttpServer.generate",
                    weight_version=args.version,
                    role="rollout",
                    pid=os.getpid(),
                    topology=target,
                ),
                generated_hash,
                args.snapshot_sha256,
            )
            write(
                args.output / "generated-batch.json",
                dict(
                    weight_version=args.version,
                    generation_source="verl.vLLMHttpServer.generate",
                    samples=generated,
                    request=request,
                    generation_seconds=elapsed,
                    response_tokens=sum(len(c["response_ids"]) for c in generated),
                ),
            )
        else:
            scores, batches = [], []
            for batch_id, batch_cases in groups(cases):
                result = await server.rlk_score_sequences(batch_cases, batch_id)
                scores.extend(result["scores"])
                batches.append(result)
            save_scores(
                args.output,
                cases,
                scores,
                dict(
                    engine="verl.vLLMHttpServer.generate",
                    weight_version=args.version,
                    role="rollout",
                    pid=os.getpid(),
                    topology=target,
                ),
                suite_hash,
                args.snapshot_sha256,
            )
            write(args.output / "rollout-batches.json", batches)
        await server.abort_all_requests(reset_prefix_cache=True, reject_request=True)
    finally:
        if hasattr(server, "engine"):
            result = server.engine.shutdown()
            if inspect.isawaitable(result):
                await result
        task = getattr(server, "_server_task", None)
        if task is not None:
            task.cancel()
        ray.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "action", choices=["actor-score", "rollout-score", "actor-update", "actor-audit", "rollout-generate"]
    )
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--snapshot-sha256", required=True)
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--next-checkpoint", type=Path)
    ap.add_argument("--batch", type=Path)
    ap.add_argument("--provenance", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    try:
        if args.action == "actor-score":
            actor_score(args)
        elif args.action == "actor-update":
            actor_update(args)
        elif args.action == "actor-audit":
            actor_audit(args)
        else:
            asyncio.run(rollout_phase(args))
        write(
            args.output / ("phase-" + os.environ.get("RANK", "driver") + ".json"),
            dict(
                status="SCORES_SAVED",
                action=args.action,
                elapsed=time.monotonic() - began,
                pid=os.getpid(),
                optimizer_updates=1 if args.action == "actor-update" else 0,
            ),
        )
    except BaseException as e:
        write(
            args.output / ("error-" + os.environ.get("RANK", "driver") + ".json"),
            dict(status="FAIL", action=args.action, error=str(e), traceback=traceback.format_exc(), pid=os.getpid()),
        )
        raise


if __name__ == "__main__":
    main()

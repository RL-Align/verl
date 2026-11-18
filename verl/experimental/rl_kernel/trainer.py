# isort: skip_file
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
# ruff: noqa: E402 -- install the version-scoped native import adapters first.
"""Opt-in synchronous verl trainer; native reward, advantage and policy loss."""

import asyncio
import json
import os
from pathlib import Path

from .runtime import prepare_worker

prepare_worker()

from . import trainer_engine  # noqa: F401

import ray
import torch
import torch.distributed as dist
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.engine_workers import ActorRolloutRefWorker

from .trainer_contract import strict_response_parity, validate_trainer_config
from .trainer_rollout import RLKTrainerReplica
from .trainer_sync import RLKPolicySync


class RLKActorWorker(ActorRolloutRefWorker):
    """Inherit native TrainingWorker and ppo_loss; add checked export only."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        from verl.workers.utils.losses import ppo_loss

        if getattr(self.loss_fn, "func", None) is not ppo_loss:
            raise RuntimeError("RLK trainer requires verl's native PPO/GRPO policy loss")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rlk_placement(self):
        from rl_engine.integrations.common.device_identity import hip_device_identity

        return dict(
            rank=dist.get_rank(),
            node_id=ray.get_runtime_context().get_node_id(),
            pid=os.getpid(),
            **hip_device_identity(),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rlk_export_policy(self, path):
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import digest

        engine = self.actor.engine
        model = engine._rlk_models[0]
        covered = {
            id(parameter) for parameters in model._rlk_checkpoint_export_parameters.values() for parameter in parameters
        }
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in covered
        ]
        if missing:
            raise RuntimeError(f"trainable parameters missing from sync export: {missing}")
        parameters, _ = engine.get_per_tensor_param(base_sync_done=True)
        manifest, weights = {}, {}
        for name, tensor in parameters:
            cpu = tensor.detach().cpu().clone()
            if not torch.isfinite(cpu).all():
                raise RuntimeError(f"nonfinite actor state: {name}")
            manifest[name] = dict(shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=digest(cpu))
            if dist.get_rank() == 0:
                weights[name] = cpu
        payload = dict(weight_version=engine.rlk_weight_version, manifest=manifest)
        if dist.get_rank() == 0:
            destination = Path(path)
            pending = destination.with_suffix(".pending")
            torch.save(dict(payload, weights=weights), pending)
            os.replace(pending, destination)
        # ONE_TO_ALL waits for both exports, including the writer, to return.
        return payload

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rlk_score_sequences(self, cases, weight_version, micro_batch_size=1, trace_label=None):
        """Score new rollout tokens with the same live native actor after sync."""
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu

        if self.actor.engine.rlk_weight_version != weight_version:
            raise RuntimeError("actor audit requested a stale weight version")
        tokens, masks = [], []
        for case in cases:
            prompt, response = case["prompt_ids"], case["response_ids"]
            if not prompt or not response:
                raise ValueError("post-update audit requires nonempty prompt and response")
            tokens.append(torch.tensor(prompt + response, dtype=torch.int64))
            masks.append(torch.tensor([0] * len(prompt) + [1] * len(response)))
        data = TensorDict(
            {
                "input_ids": torch.nested.as_nested_tensor(tokens, layout=torch.jagged),
                "loss_mask": torch.nested.as_nested_tensor(masks, layout=torch.jagged),
                "temperature": torch.ones(len(cases)),
            },
            batch_size=[len(cases)],
        )
        tu.assign_non_tensor(
            data,
            global_token_num=None,
            compute_loss=False,
            disable_auto_offload=True,
            micro_batch_size_per_gpu=micro_batch_size,
            calculate_entropy=False,
        )
        engine = self.actor.engine
        observed = []

        def actual_forward(_model, args, kwargs):
            ids = kwargs.get("input_ids")
            observed.append(
                dict(
                    shape=list(ids.shape),
                    dtype=str(ids.dtype),
                    stride=list(ids.stride()),
                    tokens=ids.detach().cpu().tolist(),
                )
            )

        handle = engine._rlk_models[0].register_forward_pre_hook(actual_forward, with_kwargs=True)
        stop_trace = None
        first_trace = os.environ.get("RLK_TRACE_FIRST_AUDIT") == "1" and not getattr(self, "_rlk_traced", False)
        if trace_label is not None and not trace_label.startswith("failure-"):
            raise ValueError("explicit actor traces are restricted to failed-audit diagnostics")
        if first_trace or trace_label:
            from examples.operators.dsv4_4l.boundary_trace import capture

            stop_trace = capture(
                engine._rlk_models[0],
                Path(self.config.actor.megatron.rlk_evidence_dir)
                / "traces"
                / (f"actor-rank-{dist.get_rank()}" + (f"-{trace_label}" if trace_label else "")),
                "actor",
            )
            self._rlk_traced = True
        try:
            output = self.actor.infer_batch(data)
        finally:
            handle.remove()
            if stop_trace is not None:
                stop_trace()
        if not observed or any(row["shape"][0] != min(micro_batch_size, len(cases)) for row in observed):
            raise RuntimeError(f"actor did not execute actual requested batch: {observed}")
        if output is None:
            return None
        scores = [
            row[len(case["prompt_ids"]) - 1 : len(case["prompt_ids"]) + len(case["response_ids"]) - 1]
            for row, case in zip(output["log_probs"].unbind(), cases, strict=True)
        ]
        return dict(weight_version=weight_version, scores=scores, forward_batches=observed)


class RLKVerlTrainer(PPOTrainerSync):
    """A narrow PP2/DP1 GRPO recipe using the ordinary upstream fit loop."""

    _enable_hybrid_replicas = False

    def __init__(self, config):
        validate_trainer_config(config)
        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        self.role_worker_mapping[Role.Actor] = ray.remote(RLKActorWorker)
        if self.config.rlk.get("multinode", False):
            from .cluster import actor_pool_manager

            self.resource_pool_manager = actor_pool_manager(
                self.resource_pool_manager, list(self.config.rlk.actor_node_ips), self.config.trainer.n_gpus_per_node
            )

    def _setup(self):
        super()._setup()
        self.checkpoint_manager.close()
        manager = self.llm_server_manager
        manager.rollout_replica_class = RLKTrainerReplica
        asyncio.run(manager._initialize_llm_servers())
        # Keep the router created by upstream setup. Do not create an orphaned
        # second router or alter the process-global vLLM replica registry.
        ray.get(
            manager.global_load_balancer.add_servers.remote(
                dict(zip(manager.server_addresses, manager.server_handles, strict=True))
            )
        )
        from .cluster import record_placement

        record_placement(
            self.config,
            self.actor_rollout_wg.rlk_placement(),
            ray.get(manager.server_handles[0].rlk_placement.remote()),
        )
        self.checkpoint_manager = RLKPolicySync(self.actor_rollout_wg, manager.server_handles, self.config.rlk.sync_dir)

    def _compute_old_log_prob(self, batch, metrics):
        import transfer_queue as tq

        batch = super()._compute_old_log_prob(batch, metrics)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=[
                "old_log_probs",
                "rollout_log_probs",
                "response_mask",
                "prompts",
                "responses",
            ],
        )
        # Preserve native lengths: DSV4's BOS is token 0, also the padded export fill.
        prompt_lengths = data["prompts"].offsets().diff().detach().cpu()
        data = data.to_padded_tensor()
        root = Path(self.config.rlk.sync_dir)
        torch.save(
            dict(data.items(), prompt_lengths=prompt_lengths),
            root / f"parity-step{self.global_steps}-raw.pt",
        )
        record = strict_response_parity(data["old_log_probs"], data["rollout_log_probs"], data["response_mask"])
        record.update(global_steps=self.global_steps, weight_version=self.checkpoint_manager.version)
        root = Path(self.config.rlk.sync_dir)
        with (root / "parity.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        metrics.update({f"rlk/{key}": record[key] for key in ("compared_tokens", "mismatch_count")})
        return batch

    def _compute_advantage(self, batch, metrics):
        import transfer_queue as tq

        batch = super()._compute_advantage(batch, metrics)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["advantages", "returns", "rm_scores", "response_mask"],
        ).to_padded_tensor()
        metadata = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["metrics", "extra_fields"],
        )

        def serial(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {str(k): serial(v) for k, v in value.items()}
            if isinstance(value, tuple | list):
                return [serial(v) for v in value]
            if hasattr(value, "to_dict"):
                return serial(value.to_dict())
            if hasattr(value, "tolist"):
                return serial(value.tolist())
            if value is None or isinstance(value, str | int | float | bool):
                return value
            raise TypeError(f"unrecorded native AgentLoop metadata: {type(value)}")

        native_agent_loop = serial(metadata)
        summaries = {}
        for name in ("advantages", "returns", "rm_scores"):
            tensor = data[name]
            if not tensor.numel() or not torch.isfinite(tensor).all():
                raise RuntimeError(f"empty/nonfinite native post-training field: {name}")
            summaries[name] = dict(
                finite=True,
                nonzero=int(torch.count_nonzero(tensor)),
                minimum=float(tensor.min()),
                maximum=float(tensor.max()),
                mean=float(tensor.float().mean()),
                std=float(tensor.float().std(unbiased=False)),
                values=tensor.tolist(),
            )
        (Path(self.config.rlk.sync_dir) / f"training-batch-step{self.global_steps}.json").write_text(
            json.dumps(
                dict(
                    global_steps=self.global_steps,
                    weight_version=self.checkpoint_manager.version,
                    algorithm="grpo",
                    native_policy_loss="verl.workers.utils.losses.ppo_loss",
                    summaries=summaries,
                    native_agent_loop=native_agent_loop,
                    rewards_per_response=data["rm_scores"].sum(-1).tolist(),
                    response_mask=data["response_mask"].tolist(),
                ),
                indent=2,
            )
        )
        if summaries["advantages"]["nonzero"] == 0:
            raise RuntimeError("all native GRPO advantages are zero; this is not an effective update")
        return batch

    def on_init_end(self):
        super().on_init_end()
        from .acceptance import audit_version

        audit_version(self)

    def on_step_end(self):
        super().on_step_end()
        from .acceptance import audit_version, stress_current_version

        audit_version(self)
        repeats = int(self.config.rlk.get("stress_final_repeats", 0))
        if self.global_steps == 3 and repeats:
            stress_current_version(self, repeats)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        def encode(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(type(value))

        (Path(self.config.rlk.sync_dir) / f"metrics-step{global_steps}.json").write_text(
            json.dumps(
                dict(
                    global_steps=global_steps,
                    metrics=metrics,
                    timing_seconds=timing_raw,
                    rollout_gpus=2,
                    total_job_gpus=4,
                    includes_audit=True,
                ),
                indent=2,
                default=encode,
            )
        )

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
# ruff: noqa: E402 -- native adapters must be installed before importing verl.
"""Native verl loss ABI backed by RLK's DSV4 operator boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path

from .runtime import prepare_worker

prepare_worker()

import torch
from rl_engine.integrations.orchestrators.verl.native_engine import (
    RLKNativeMcoreEngineConfig,
    RLKNativeMegatronEngine,
)
from verl.workers.engine.base import EngineRegistry

from .trainer_contract import logical_cases


def bind_actor_operators(model):
    from rl_engine.integrations.engines.train.megatron.native_combine import bind_native_combine
    from rl_engine.integrations.engines.train.megatron.native_compressor import (
        bind_native_compressor,
    )
    from rl_engine.integrations.engines.train.megatron.native_compressor_projection import (
        bind_compressor_projection,
    )
    from rl_engine.integrations.engines.train.megatron.native_hash_route import (
        bind_native_hash_route,
    )
    from rl_engine.integrations.engines.train.megatron.native_inverse_rope import (
        bind_native_inverse_rope,
    )
    from rl_engine.integrations.engines.train.megatron.native_learned_route import (
        bind_native_learned_route,
    )
    from rl_engine.integrations.engines.train.megatron.native_router_projection import (
        bind_native_router_projection,
    )
    from rl_engine.integrations.engines.train.megatron.native_sparse import bind_native_sparse
    from rl_engine.integrations.engines.train.megatron.native_w4a16 import bind_native_w4a16

    for layer in model.decoder.layers[::2]:
        attention = layer.inner_layer.self_attention
        bind_native_sparse(attention.core_attention)
        bind_native_inverse_rope(attention)
        if getattr(attention.core_attention, "compressor", None) is not None:
            bind_compressor_projection(attention.core_attention.compressor)
            bind_native_compressor(attention.core_attention.compressor, attention._rlk_qkv_preprocess)
        if getattr(attention.core_attention, "indexer", None) is not None:
            bind_compressor_projection(attention.core_attention.indexer.compressor)
    for layer in model.decoder.layers[1::2]:
        mlp = layer.inner_layer.mlp
        bind_native_router_projection(mlp.router)
        binder = bind_native_hash_route if mlp.router.is_hash_layer else bind_native_learned_route
        binder(mlp.router)
        bind_native_w4a16(mlp)
        bind_native_combine(mlp)


@dataclass
class RLKVerlMcoreEngineConfig(RLKNativeMcoreEngineConfig):
    strategy: str = "rlk_verl_megatron"
    rlk_allow_right_padding: bool = True
    rlk_evidence_dir: str = ""
    rlk_require_update_forward_parity: bool = False

    def __post_init__(self):
        if self.strategy != "rlk_verl_megatron":
            raise ValueError("expected the explicit rlk_verl_megatron registration")
        object.__setattr__(self, "strategy", "rlk_megatron")
        try:
            super().__post_init__()
        finally:
            object.__setattr__(self, "strategy", "rlk_verl_megatron")


@EngineRegistry.register("language_model", "rlk_verl_megatron")
class RLKVerlMegatronEngine(RLKNativeMegatronEngine):
    def initialize(self):
        super().initialize()
        for model in self._rlk_models:
            bind_actor_operators(model)
        from examples.operators.dsv4_4l.distributed_evidence import PipelineBoundaryEvidence

        self._rlk_boundary = PipelineBoundaryEvidence(self._rlk_models[0], dp_rank=0)
        self._rlk_previous_optimizer = None
        if self.engine_config.rlk_evidence_dir:
            import os

            import torch.distributed as dist
            from rl_engine.integrations.common.device_identity import hip_device_identity

            root = Path(self.engine_config.rlk_evidence_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / f"actor-rank-{dist.get_rank()}-identity.json").write_text(
                json.dumps(
                    dict(
                        rank=dist.get_rank(),
                        actual_model_class=type(self._rlk_models[0]).__module__
                        + "."
                        + type(self._rlk_models[0]).__qualname__,
                        actual_model_source=__import__("inspect").getfile(type(self._rlk_models[0])),
                        trainable_inventory={
                            name: dict(shape=list(p.shape), dtype=str(p.dtype), elements=p.numel())
                            for name, p in self._rlk_models[0].named_parameters()
                            if p.requires_grad
                        },
                        pid=os.getpid(),
                        source_identity=__import__(
                            "verl.experimental.rl_kernel.runtime", fromlist=["process_identity"]
                        ).process_identity("actor"),
                        device_identity=hip_device_identity(),
                        original_layer_ids=list(self._rlk_models[0]._rlk_checkpoint_stage.layer_ids),
                    ),
                    indent=2,
                )
            )

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        from verl.utils import tensordict_utils as tu

        batch = next(batch_iter)
        self.rlk_logical_cases = logical_cases(batch["input_ids"])
        # Native CSA requires a window-sized workspace even for short prompts.
        # Keep real tokens/masks unchanged and declare only BSHD right padding.
        forced = tu.get_non_tensor_data(batch, key="forced_max_seqlen", default=0) or 0
        actual_max = max(len(row) for row in batch["input_ids"].unbind())
        tu.assign_non_tensor(
            batch,
            forced_max_seqlen=max(int(forced), actual_max, int(getattr(self.tf_config, "csa_window_size", 1))),
        )
        return super().forward_step(iter((batch,)), model, logits_processor_func, postprocess_micro_batch_func)

    def _lm_head_logits_processor(self, logits, label, temperature, **kwargs):
        # Padded temperature entries are zero; only nonpadding entries may be 1.
        if not torch.all((temperature == 1) | (temperature == 0)):
            raise ValueError("RLK verl raw-score ABI requires temperature=1")
        if any(
            kwargs.get(key, False)
            for key in (
                "distillation_use_topk",
                "distillation_only",
                "score_centering",
                "calculate_sum_pi_squared",
            )
        ):
            raise ValueError("unsupported transformed policy score in RLK verl engine")
        result = super()._lm_head_logits_processor(logits, label, temperature, **kwargs)
        # Preserve verl's PPO/GRPO loss and its native next-token shift/masks.
        # Only the selected-score provider changes, analogous to the Vime ABI.
        result["log_probs"] = result["raw_log_probs"]
        return result

    def forward_backward_batch(self, data, loss_function=None, forward_only=False):
        if forward_only:
            return super().forward_backward_batch(data, loss_function, forward_only)
        import inspect

        import torch.distributed as dist
        from verl.workers.utils.losses import ppo_loss

        if getattr(loss_function, "func", None) is not ppo_loss:
            raise RuntimeError("training must use the native verl PPO policy loss")
        calls = []

        def observed_loss(*args, **kwargs):
            update_parity = None
            if self.engine_config.rlk_require_update_forward_parity:
                from verl.workers.utils.padding import no_padding_2_padding
                from .trainer_contract import strict_response_parity

                bound = inspect.signature(loss_function).bind(*args, **kwargs).arguments
                model_output, batch = bound["model_output"], bound["data"]
                current = no_padding_2_padding(model_output["log_probs"], batch)
                original = batch.select("old_log_probs", "response_mask").to_padded_tensor()
                if not self.engine_config.rlk_evidence_dir:
                    raise RuntimeError("strict update-forward checking requires retained raw evidence")
                torch.save(
                    dict(
                        current=current.detach().cpu(),
                        old=original["old_log_probs"].detach().cpu(),
                        response_mask=original["response_mask"].detach().cpu(),
                        weight_version=self.rlk_weight_version,
                        input_token_rows=[row.detach().cpu().tolist() for row in batch["input_ids"].unbind()],
                    ),
                    Path(self.engine_config.rlk_evidence_dir)
                    / f"update-forward-rank{dist.get_rank()}-v{self.rlk_weight_version + 1}-micro{len(calls)}.pt",
                )
                update_parity = strict_response_parity(current, original["old_log_probs"], original["response_mask"])
            result = loss_function(*args, **kwargs)
            loss = result[0]
            calls.append(
                dict(
                    callable=ppo_loss.__module__ + "." + ppo_loss.__name__,
                    source=inspect.getfile(ppo_loss),
                    value=float(loss.detach()),
                    finite=bool(torch.isfinite(loss)),
                    requires_grad=loss.requires_grad,
                    update_forward_parity=update_parity,
                )
            )
            return result

        self._rlk_boundary.begin(f"update-v{self.rlk_weight_version + 1}")
        self._rlk_autograd_receipts = {}
        handles = []
        for model in self._rlk_models:
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue

                def received(gradient, name=name):
                    if gradient is not None:
                        self._rlk_autograd_receipts.setdefault(name, []).append(
                            dict(
                                shape=list(gradient.shape),
                                dtype=str(gradient.dtype),
                                finite=bool(torch.isfinite(gradient).all()),
                                nonzero=bool(torch.count_nonzero(gradient)),
                            )
                        )
                    return gradient

                handles.append(parameter.register_hook(received))
        try:
            result = super().forward_backward_batch(data, observed_loss, False)
        finally:
            for handle in handles:
                handle.remove()
        if self.engine_config.rlk_evidence_dir:
            (
                Path(self.engine_config.rlk_evidence_dir)
                / f"native-loss-rank{dist.get_rank()}-v{self.rlk_weight_version + 1}.json"
            ).write_text(json.dumps(calls, indent=2))
        return result

    def optimizer_step(self):
        import torch.distributed as dist
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import digest

        parameters = {
            name: parameter
            for model in self._rlk_models
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        from examples.operators.dsv4_4l.distributed_evidence import tensor_record
        from examples.operators.dsv4_4l.optimizer_evidence import (
            optimizer_snapshot,
            verify_optimizer_transition,
        )

        optimizer_before = optimizer_snapshot(self)
        before = {name: digest(parameter) for name, parameter in parameters.items()}
        gradients = []
        gradient_records = {}
        for name, parameter in parameters.items():
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is not None:
                gradients.append(gradient)
                gradient_records[name] = tensor_record(gradient)
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        finite = finite and all(row["finite"] for rows in self._rlk_autograd_receipts.values() for row in rows)
        missing_effective = [
            name
            for name, record in gradient_records.items()
            if record["nonzero"] and not self._rlk_autograd_receipts.get(name)
        ]
        if missing_effective:
            raise RuntimeError(f"effective optimizer gradients lack actual autograd receipt: {missing_effective}")
        if not finite:
            raise RuntimeError("nonfinite actor gradients before the native optimizer step")
        nonzero = any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
        if not gradients or not nonzero:
            raise RuntimeError("native PPO produced no effective gradient on this actor PP stage")
        result = super().optimizer_step()
        optimizer_after = optimizer_snapshot(self)
        verify_optimizer_transition(
            optimizer_before,
            optimizer_after,
            version=self.rlk_weight_version,
            previous=self._rlk_previous_optimizer,
        )
        self._rlk_previous_optimizer = optimizer_after
        changed = [name for name, parameter in parameters.items() if digest(parameter) != before[name]]
        if not changed:
            raise RuntimeError("native optimizer did not change any trainable parameter on this PP stage")
        if self.engine_config.rlk_evidence_dir:
            root = Path(self.engine_config.rlk_evidence_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / f"actor-rank-{dist.get_rank()}-v{self.rlk_weight_version}.json").write_text(
                json.dumps(
                    dict(
                        weight_version=self.rlk_weight_version,
                        rank=dist.get_rank(),
                        optimizer_before=optimizer_before,
                        optimizer_after=optimizer_after,
                        optimizer_continuity=True,
                        pipeline_transfers=self._rlk_boundary.records,
                        gradient_records=gradient_records,
                        autograd_receipts=self._rlk_autograd_receipts,
                        inactive_autograd_parameters=[
                            name for name in parameters if name not in self._rlk_autograd_receipts
                        ],
                        parameter_before=before,
                        parameter_after={name: digest(p) for name, p in parameters.items()},
                        trainable_parameters=len(parameters),
                        gradient_tensors=len(gradients),
                        gradients_finite=finite,
                        gradients_nonzero=nonzero,
                        changed_parameters=changed,
                    ),
                    indent=2,
                )
            )
        return result

    def lr_scheduler_step(self):
        """Include the native scheduler's real changes in the step-boundary snapshot."""
        import inspect
        import torch.distributed as dist
        from examples.operators.dsv4_4l.optimizer_evidence import optimizer_snapshot

        before = optimizer_snapshot(self)
        if (
            self._rlk_previous_optimizer is None
            or before["state_digest"] != self._rlk_previous_optimizer["state_digest"]
        ):
            raise RuntimeError("optimizer state changed before the native scheduler")
        result = super().lr_scheduler_step()
        after = optimizer_snapshot(self)
        if before["state_tensors"] != after["state_tensors"]:
            raise RuntimeError("native scheduler unexpectedly changed optimizer moments")
        changes = {
            key: dict(before=value, after=after["state_scalars"].get(key))
            for key, value in before["state_scalars"].items()
            if value != after["state_scalars"].get(key)
        }
        if any(key.rsplit("/", 1)[-1] not in ("lr", "weight_decay") for key in changes):
            raise RuntimeError(f"unexpected optimizer scalar change in scheduler: {changes}")
        self._rlk_previous_optimizer = after
        if self.engine_config.rlk_evidence_dir:
            path = (
                Path(self.engine_config.rlk_evidence_dir)
                / f"actor-rank-{dist.get_rank()}-v{self.rlk_weight_version}.json"
            )
            record = json.loads(path.read_text())
            record["optimizer_after_optimizer_step"] = record["optimizer_after"]
            record["optimizer_after"] = after
            record["native_scheduler"] = dict(
                source=inspect.getfile(super().lr_scheduler_step),
                changes=changes,
                moments_unchanged=True,
            )
            path.write_text(json.dumps(record, indent=2))
        return result

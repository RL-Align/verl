# Copyright 2026 RL-Kernel Contributors
# Licensed under the Apache License, Version 2.0

"""Opt-in verl boundary for RL-Kernel's shared Megatron operator scheduler."""

import os
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu


def enabled() -> bool:
    return os.environ.get("VERL_RL_KERNEL", "0") == "1"


@dataclass(frozen=True)
class VocabPartition:
    local_start: int
    local_size: int
    real_size: int
    padded_size: int


@dataclass(frozen=True)
class TokenLayout:
    rank: int
    world_size: int
    layout: str


@dataclass(frozen=True)
class LinearProjection:
    weight: torch.Tensor
    bias: torch.Tensor | None


@dataclass(frozen=True)
class LinearLogpContext:
    hidden: torch.Tensor
    projection: LinearProjection
    vocab_partition: VocabPartition
    local_logits: torch.Tensor
    reuse_local_logits: bool = True


@dataclass(frozen=True)
class LinearLogpRequest:
    logits: torch.Tensor
    target_ids: torch.Tensor
    tensor_parallel_group: Any
    token_layout: TokenLayout
    context: LinearLogpContext
    temperature: torch.Tensor
    metadata: dict[str, Any]
    with_entropy: bool = False
    with_entropy_grad: bool = False
    log_prob_keep_mask: torch.Tensor | None = None


def install(engine_config: Any) -> None:
    if not enabled():
        return
    if engine_config.use_fused_kernels or not engine_config.use_remove_padding:
        raise RuntimeError("VERL_RL_KERNEL requires THD remove-padding and unfused LM-head")
    from rl_engine.integrations.ablation import Implementation, integration_plan_from_environment
    from rl_engine.integrations.megatron_runtime import initialize_from_environment
    from rl_engine.integrations.canonical_cp import install_verl

    plan = integration_plan_from_environment()
    if any(plan.implementation_for(op, "training") is not Implementation.RL_KERNEL for op in ("attention", "ffn", "logp")):
        raise RuntimeError("VERL_RL_KERNEL requires R/R for Attention, FFN and Logp")
    initialize_from_environment(patch_output_layer=False, canonical_cp_installer=install_verl)


class _GatherSequenceParallelHidden(torch.autograd.Function):
    """Gather TP-owned token rows; projection backward already reduces dgrad over TP."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.world = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)
        if ctx.world == 1:
            return hidden
        result = torch.empty(
            (hidden.shape[0] * ctx.world, *hidden.shape[1:]), device=hidden.device, dtype=hidden.dtype
        )
        dist.all_gather_into_tensor(result, hidden.contiguous(), group=group)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.chunk(ctx.world, dim=0)[ctx.rank].contiguous(), None


def capture_lm_head_input(output_layer: Any) -> None:
    if not enabled():
        return
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from rl_engine.integrations.megatron_runtime import _DeterministicTPOutputProjection

    if type(output_layer) is not ColumnParallelLinear:
        raise RuntimeError("strict RL-Kernel LM head requires Megatron's ColumnParallelLinear")
    if output_layer.gather_output or output_layer.disable_grad_reduce:
        raise RuntimeError("strict RL-Kernel LM head requires local TP logits and ordinary dgrad")

    def forward(module: Any, input_: torch.Tensor, weight=None, runtime_gather_output=None):
        if runtime_gather_output or module.explicit_expert_comm:
            raise RuntimeError("strict RL-Kernel LM head does not support gathered/expert logits")
        projection_weight = module.weight if weight is None else weight
        if projection_weight is None:
            raise RuntimeError("strict RL-Kernel LM head requires a projection weight")
        bias = module.bias if not module.skip_bias_add else None
        hidden = _GatherSequenceParallelHidden.apply(input_, module.tp_group) if module.sequence_parallel else input_
        output = _DeterministicTPOutputProjection.apply(hidden, projection_weight, bias, module.tp_group)
        module._verl_rl_kernel_hidden = hidden
        module._verl_rl_kernel_weight = projection_weight
        module._rl_kernel_local_logits = output
        return output, (module.bias if module.skip_bias_add else None)

    output_layer.forward = MethodType(forward, output_layer)


def compute_logp(
    output_layer: Any,
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: torch.Tensor,
    *,
    real_vocab_size: int,
    cp_layout: str,
    with_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from rl_engine.integrations.vime.linear_logp_provider import provider

    hidden = getattr(output_layer, "_verl_rl_kernel_hidden", None)
    output_layer._verl_rl_kernel_hidden = None
    projected = getattr(output_layer, "_rl_kernel_local_logits", None)
    output_layer._rl_kernel_local_logits = None
    if not isinstance(hidden, torch.Tensor) or not isinstance(projected, torch.Tensor):
        raise RuntimeError("RL-Kernel LM-head hidden rows were not captured for this forward")
    if projected is not logits:
        if projected.ndim != 3 or logits.ndim != 3 or not torch.equal(projected.transpose(0, 1), logits):
            raise RuntimeError("RL-Kernel LM-head logits are not the active projection output")
        hidden = hidden.transpose(0, 1)
    if hidden.shape[:-1] != logits.shape[:-1]:
        raise RuntimeError("RL-Kernel LM-head hidden rows do not match logits")
    if not isinstance(getattr(output_layer, "_verl_rl_kernel_weight", None), torch.Tensor):
        raise RuntimeError("RL-Kernel LM-head projection weight was not captured")
    if logits.dtype != torch.bfloat16:
        raise RuntimeError(
            f"strict RL-Kernel logp needs BF16 local logits, got logits={logits.dtype}, "
            f"projected={projected.dtype}, weight={output_layer._verl_rl_kernel_weight.dtype}, "
            f"shape={tuple(logits.shape)}"
        )
    if labels.shape != logits.shape[:-1] or temperature.shape != labels.shape:
        raise RuntimeError("RL-Kernel token/temperature rows do not match the local logits")
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    if cp_layout not in ({"single"} if cp_size == 1 else {"zigzag", "allgather"}):
        raise RuntimeError(f"unsupported RL-Kernel CP layout: {cp_layout}")
    local_size = logits.shape[-1]
    request = LinearLogpRequest(
        logits=logits.reshape(-1, local_size),
        target_ids=labels.reshape(-1),
        tensor_parallel_group=tp_group,
        token_layout=TokenLayout(cp_rank, cp_size, cp_layout),
        context=LinearLogpContext(
            hidden=hidden.reshape(-1, hidden.shape[-1]),
            projection=LinearProjection(output_layer._verl_rl_kernel_weight, getattr(output_layer, "bias", None)),
            vocab_partition=VocabPartition(tp_rank * local_size, local_size, real_vocab_size, local_size * tp_size),
            local_logits=logits.reshape(-1, local_size),
        ),
        temperature=temperature.reshape(-1),
        metadata={"tp_rank": tp_rank, "tp_world_size": tp_size, "complete_sampling_support": True},
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy,
    )
    result = provider(request)
    if not isinstance(result.logp, torch.Tensor) or result.logp.numel() != labels.numel():
        raise RuntimeError("RL-Kernel returned an invalid logp tensor")
    if logits.requires_grad and not result.logp.requires_grad:
        raise RuntimeError("RL-Kernel detached the training logp graph")
    if with_entropy and (result.entropy is None or result.entropy.numel() != labels.numel()):
        raise RuntimeError("RL-Kernel did not return per-token entropy")
    return result.logp.reshape_as(labels), (result.entropy.reshape_as(labels) if with_entropy else None)

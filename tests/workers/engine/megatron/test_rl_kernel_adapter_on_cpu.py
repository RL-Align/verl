from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.megatron import rl_kernel


def test_native_path_does_not_load_rl_kernel(monkeypatch):
    monkeypatch.delenv("VERL_RL_KERNEL", raising=False)
    rl_kernel.install(SimpleNamespace(use_fused_kernels=False, use_remove_padding=True))


def test_padded_cp_indices_respect_tp_alignment():
    from rl_engine.integrations.canonical_cp import padded_gather_indices

    # Each sequence has 16 padded rows, CP2 owns four rows from each end.
    assert padded_gather_indices((5, 3), (16, 16), 16, 2) == [0, 1, 2, 3, 16, 8, 9, 10]
    with pytest.raises(ValueError, match="rows differ"):
        padded_gather_indices((5,), (16,), 16, 2)


def test_strict_path_rejects_incompatible_layout(monkeypatch):
    monkeypatch.setenv("VERL_RL_KERNEL", "1")
    with pytest.raises(RuntimeError, match="THD remove-padding"):
        rl_kernel.install(SimpleNamespace(use_fused_kernels=True, use_remove_padding=True))


def test_strict_logp_requires_current_projection(monkeypatch):
    from rl_engine.integrations.vime import linear_logp_provider

    monkeypatch.setattr(linear_logp_provider, "provider", lambda request: pytest.fail("provider called"))
    layer = SimpleNamespace(
        _verl_rl_kernel_hidden=torch.ones(2, 1, 4),
        _rl_kernel_local_logits=torch.empty(2, 1, 8),
    )
    with pytest.raises(RuntimeError, match="active projection"):
        rl_kernel.compute_logp(
            layer,
            torch.zeros(2, 1, 8),
            torch.zeros(2, 1, dtype=torch.long),
            torch.ones(2, 1),
            real_vocab_size=8,
            cp_layout="single",
            with_entropy=False,
        )


def test_sequence_parallel_gather_backward_selects_local_rows(monkeypatch):
    monkeypatch.setattr(rl_kernel.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(rl_kernel.dist, "get_rank", lambda group: 1)

    def gather(result, hidden, group):
        result[: hidden.shape[0]] = -hidden
        result[hidden.shape[0] :] = hidden

    monkeypatch.setattr(rl_kernel.dist, "all_gather_into_tensor", gather)
    hidden = torch.ones(2, 1, 4, requires_grad=True)
    gathered = rl_kernel._GatherSequenceParallelHidden.apply(hidden, None)
    assert gathered.shape == (4, 1, 4)
    gathered.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))


def test_strict_logp_preserves_grad_and_row_shape(monkeypatch):
    from rl_engine.integrations.vime import linear_logp_provider

    monkeypatch.setattr(rl_kernel.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(rl_kernel.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(rl_kernel.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rl_kernel.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(rl_kernel.mpu, "get_context_parallel_world_size", lambda: 1)
    logits = torch.ones(2, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    layer = SimpleNamespace(
        _verl_rl_kernel_hidden=torch.ones(2, 1, 4),
        _verl_rl_kernel_weight=torch.ones(8, 4),
        _rl_kernel_local_logits=logits,
        bias=None,
    )

    def provider(request):
        assert request.context.reuse_local_logits
        assert request.context.vocab_partition.padded_size == 8
        return SimpleNamespace(logp=request.logits.sum(-1).unsqueeze(-1), entropy=None)

    monkeypatch.setattr(linear_logp_provider, "provider", provider)
    logp, entropy = rl_kernel.compute_logp(
        layer,
        logits,
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1),
        real_vocab_size=8,
        cp_layout="single",
        with_entropy=False,
    )
    assert logp.shape == (2, 1)
    assert entropy is None
    logp.sum().backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))


def test_strict_logp_accepts_megatron_batch_first_output(monkeypatch):
    from rl_engine.integrations.vime import linear_logp_provider

    monkeypatch.setattr(rl_kernel.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(rl_kernel.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(rl_kernel.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rl_kernel.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(rl_kernel.mpu, "get_context_parallel_world_size", lambda: 1)
    hidden = torch.arange(12, dtype=torch.float).reshape(3, 1, 4)
    projected = torch.ones(3, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    logits = projected.transpose(0, 1).contiguous()
    layer = SimpleNamespace(
        _verl_rl_kernel_hidden=hidden,
        _verl_rl_kernel_weight=torch.ones(8, 4),
        _rl_kernel_local_logits=projected,
        bias=None,
    )

    def provider(request):
        assert request.context.hidden.shape == (3, 4)
        torch.testing.assert_close(request.context.hidden, hidden[:, 0])
        return SimpleNamespace(logp=request.logits.sum(-1), entropy=None)

    monkeypatch.setattr(linear_logp_provider, "provider", provider)
    logp, _ = rl_kernel.compute_logp(
        layer, logits, torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3),
        real_vocab_size=8, cp_layout="single", with_entropy=False,
    )
    assert logp.shape == (1, 3)
    logp.sum().backward()
    torch.testing.assert_close(projected.grad, torch.ones_like(projected))

# SPDX-License-Identifier: Apache-2.0
"""CPU recovery contracts only, never DSV4 GPU or GRPO acceptance."""

from types import SimpleNamespace as NS

import pytest
import torch

from verl.experimental.rl_kernel.phase_checkpoint import capture, restore, section_digests, state_digest


def engine():
    model = torch.nn.Linear(3, 2, dtype=torch.float32)
    model.register_buffer("packed_original", torch.arange(12, dtype=torch.uint8))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return NS(
        _rlk_models=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        rlk_weight_version=0,
        _rlk_previous_optimizer=None,
    )


def step(obj):
    obj.optimizer.zero_grad()
    obj._rlk_models[0](torch.tensor([[1.0, 2.0, 3.0]])).square().sum().backward()
    obj.optimizer.step()
    obj.lr_scheduler.step()
    obj.rlk_weight_version += 1


def test_three_actual_adam_updates_survive_process_style_reconstruction_exactly():
    torch.manual_seed(8)
    uninterrupted = engine()
    resumed = engine()
    resumed._rlk_models[0].load_state_dict(uninterrupted._rlk_models[0].state_dict())
    provenance = dict(model="CPU-test-only", topology="same-rank-layout", source="test")
    for version in range(1, 4):
        step(uninterrupted)
        step(resumed)
        payload = capture(resumed, provenance, capture_cuda=False)
        fresh = engine()
        fresh.rlk_weight_version = version
        parameter_ids = [id(p) for p in fresh._rlk_models[0].parameters()]
        restore(fresh, payload, provenance, restore_cuda=False)
        assert parameter_ids == [id(p) for p in fresh._rlk_models[0].parameters()]
        assert state_digest(fresh._rlk_models[0].state_dict()) == state_digest(
            uninterrupted._rlk_models[0].state_dict()
        )
        assert state_digest(fresh.optimizer.state_dict()) == state_digest(uninterrupted.optimizer.state_dict())
        assert state_digest(fresh.lr_scheduler.state_dict()) == state_digest(uninterrupted.lr_scheduler.state_dict())
        resumed = fresh


def test_section_digests_exclude_provenance_and_process_identity():
    torch.manual_seed(3)
    first, second = engine(), engine()
    second._rlk_models[0].load_state_dict(first._rlk_models[0].state_dict())
    step(first)
    step(second)
    # Same training state, different group and process evidence (ids, data pointers).
    first._rlk_previous_optimizer = dict(pid=1, optimizer_id=id(first.optimizer), data_ptr=11)
    second._rlk_previous_optimizer = dict(pid=2, optimizer_id=id(second.optimizer), data_ptr=22)
    a = capture(first, dict(group="N1", source="test"), capture_cuda=False)
    b = capture(second, dict(group="N2", source="test"), capture_cuda=False)
    b["rng"] = a["rng"]  # both groups restore the same seeded RNG; this test isolates metadata
    assert state_digest(a) != state_digest(b)
    assert section_digests(a) == section_digests(b)
    assert set(section_digests(a)) == {
        "model_state_digest",
        "optimizer_state_digest",
        "scheduler_state_digest",
        "rng_state_digest",
    }
    step(second)
    changed = section_digests(capture(second, dict(group="N2", source="test"), capture_cuda=False))
    assert changed["model_state_digest"] != section_digests(a)["model_state_digest"]
    assert changed["optimizer_state_digest"] != section_digests(a)["optimizer_state_digest"]


def test_recovery_requires_the_same_version_source_and_topology():
    original = engine()
    step(original)
    saved = capture(original, dict(source="first"), capture_cuda=False)
    fresh = engine()
    with pytest.raises(ValueError, match="saved weight version"):
        restore(fresh, saved, dict(source="first"), restore_cuda=False)
    fresh.rlk_weight_version = 1
    with pytest.raises(ValueError, match="provenance"):
        restore(fresh, saved, dict(source="different"), restore_cuda=False)


def test_raw_scores_reject_rounding_nonfinite_and_preserve_signed_zero():
    import struct

    from verl.experimental.rl_kernel.phase_runtime import raw_bytes

    assert raw_bytes([0.0, -0.0]) == struct.pack("<ff", 0.0, -0.0)
    for value in [float("nan"), float("inf"), float("-inf"), 0.1]:
        with pytest.raises(ValueError):
            raw_bytes([value])
    with pytest.raises(ValueError):
        raw_bytes(torch.ones(1, dtype=torch.float64))

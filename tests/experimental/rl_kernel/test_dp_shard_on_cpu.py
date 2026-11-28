# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for lockstep DP sharding of the phase actors, never GPU/EP acceptance."""

import pytest

from verl.experimental.rl_kernel import dp_shard
from verl.experimental.rl_kernel.dp_shard import (
    gather_in_order,
    merge_gathered,
    require_even,
    shard_cases,
    shard_indices,
)


def simulate(n, dp, pp_ranks=2):
    """World payloads as all_gather_object would see them: PP rank 0 owns nothing."""
    cases = [f"c{i}" for i in range(n)]
    world = []
    for _pp in range(pp_ranks - 1):
        world.extend([None] * dp)
    for d in range(dp):
        local, _ = shard_cases(cases, d, dp)
        world.append((d, [f"score-{c}" for c in local]))
    return cases, world


def test_dp1_is_identity():
    cases = ["a", "b", "c"]
    local, indices = shard_cases(cases, 0, 1)
    assert local == cases and indices == [0, 1, 2]
    assert merge_gathered([None, (0, [f"s{c}" for c in cases])], 3, 1) == ["sa", "sb", "sc"]


@pytest.mark.parametrize("n,dp", [(4, 4), (128, 4), (5, 4), (3, 4), (1, 4), (8, 4), (7, 2)])
def test_lockstep_and_order_restored(n, dp):
    cases, world = simulate(n, dp)
    rounds = -(-n // dp)
    for d in range(dp):
        local, indices = shard_cases(cases, d, dp)
        assert len(local) == len(indices) == rounds
    assert merge_gathered(world, n, dp) == [f"score-{c}" for c in cases]


def test_assignment_is_strided_and_pads_with_duplicate():
    cases = ["a", "b", "c", "d", "e"]
    assert shard_indices(5, 0, 4) == [0, 4]
    assert shard_indices(5, 1, 4) == [1, None]
    local, _ = shard_cases(cases, 3, 4)
    assert local == ["d", "d"]  # padding duplicates the rank's last real case
    local, indices = shard_cases(["a", "b"], 3, 4)
    assert local == ["a"] and indices == [None]  # rank with no real case duplicates case 0


def test_padding_outputs_are_dropped():
    # Padded slots return a distinct marker; it must never reach the merged output.
    world = [(0, ["r0", "r4"]), (1, ["r1", "PAD"]), (2, ["r2", "PAD"]), (3, ["r3", "PAD"])]
    assert merge_gathered(world, 5, 4) == ["r0", "r1", "r2", "r3", "r4"]


def test_merge_rejects_missing_or_duplicate_owners():
    with pytest.raises(RuntimeError, match="cover DP ranks"):
        merge_gathered([(0, ["a"]), None], 2, 2)
    with pytest.raises(RuntimeError, match="more than one"):
        merge_gathered([(0, ["a"]), (0, ["a"])], 2, 2)
    with pytest.raises(RuntimeError, match="slot count"):
        merge_gathered([(0, ["a", "b"]), (1, ["c"])], 2, 2)


def test_require_even():
    require_even(128, 4)
    require_even(4, 1)
    with pytest.raises(ValueError, match="not divisible"):
        require_even(6, 4)


def test_gather_in_order_uses_world_all_gather(monkeypatch):
    import torch.distributed as dist

    cases, world = simulate(6, 4)
    me = len(world) - 1  # a last-stage rank

    def fake_all_gather(output, payload):
        assert payload == world[me]
        output[:] = world

    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: len(world))
    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather)
    local, _ = shard_cases(cases, 3, 4)
    got = gather_in_order([f"score-{c}" for c in local], 6, 3, 4, True)
    assert got == [f"score-{c}" for c in cases]
    assert dp_shard.gather_in_order is gather_in_order


def test_non_owner_contributes_none(monkeypatch):
    import torch.distributed as dist

    seen = {}

    def fake_all_gather(output, payload):
        seen["payload"] = payload
        output[:] = [None, (0, ["x"])]

    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)
    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather)
    assert gather_in_order([], 1, 0, 1, False) == ["x"]
    assert seen["payload"] is None

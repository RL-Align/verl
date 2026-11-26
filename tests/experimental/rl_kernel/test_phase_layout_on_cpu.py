# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for PP2/DP1 and PP2xDP4xEP4 rank layout checks (synthetic receipts, no GPU)."""

import json
import sys
from types import SimpleNamespace as NS

import pytest

from verl.experimental.rl_kernel import phase_checkpoint as pc

PCI = [f"0000:{i:02x}:00.0" for i in range(8)]


def _actor_files(root, n, dp, ep, *, bad_dp=False):
    for r in range(n):
        ident = dict(
            rank=r,
            original_layers=[0, 1, 2, 3] if r < n // 2 else [4, 5, 6],
            pci_bus_id=PCI[r].upper(),
            dp_rank=r % dp,
            dp_size=dp,
            ep_rank=r % dp if ep > 1 else 0,
            ep_size=ep,
        )
        if bad_dp and r == 5:
            ident["dp_rank"] = 0
        (root / f"actor-rank-{r}.json").write_text(json.dumps(ident))


def _run(tmp_path, layout, **kw):
    acceptance = pytest.importorskip("verl.experimental.rl_kernel.phase_acceptance")
    n, dp, ep = layout
    actor = tmp_path / "actor"
    actor.mkdir()
    _actor_files(actor, n, dp, ep, **kw)
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(dict(id="X", actor=dict(pp=2, tp=1, ep=ep, etp=1, dp=dp, gpus=n), rollout=dict(tp=8)))
    )
    alloc = tmp_path / "alloc.json"
    alloc.write_text(json.dumps([dict(pci_bus_id=p) for p in PCI]))
    args = NS(
        profile=profile,
        allocation=alloc,
        actor=actor,
        rollout=tmp_path / "missing-rollout",
        policy=tmp_path / "missing.pt",
    )
    return acceptance.verify(args)


@pytest.mark.parametrize("layout", [(2, 1, 1), (8, 4, 4)])
def test_actor_layout_passes_up_to_rollout_files(tmp_path, layout):
    # Layout checks pass; failure is the first later step (missing rollout receipts).
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, layout)


def test_actor_dp_identity_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="dp_rank"):
        _run(tmp_path, (8, 4, 4), bad_dp=True)


def _megatron(monkeypatch, pp, dp, ep):
    mpu = NS(
        get_pipeline_model_parallel_rank=lambda: pp,
        get_data_parallel_rank=lambda: dp,
        get_expert_model_parallel_rank=lambda: ep,
    )
    core = NS(parallel_state=mpu)
    monkeypatch.setitem(sys.modules, "megatron", NS(core=core))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)


def test_rank_coordinates_recorded_and_compared(monkeypatch):
    model = NS(_rlk_expert_placement=dict(ep_size=4, ep_rank=2, experts={0: (8, 9)}))
    engine = NS(_rlk_models=[model])
    _megatron(monkeypatch, 1, 2, 2)
    coords = pc.rank_coordinates(engine)
    assert coords["pp_rank"] == 1 and coords["dp_rank"] == 2 and coords["ep_rank"] == 2
    assert coords["expert_placement"] == [dict(ep_size=4, ep_rank=2, experts={"0": [8, 9]})]
    assert json.loads(json.dumps(coords)) == coords
    # No placement and no megatron-recorded key: still well formed.
    assert "expert_placement" not in pc.rank_coordinates(NS(_rlk_models=[NS()]))

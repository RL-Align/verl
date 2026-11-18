# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; these do not emulate model or RCCL execution."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/rl_kernel"
PACKAGE = "_rlk_cluster_contract_tests"
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = pkg


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cluster = load("cluster_contract")
contract = load("trainer_contract")


def config(actor=("10.0.0.1", "10.0.0.2"), rollout=("10.0.0.1", "10.0.0.2")):
    return dict(
        rlk=dict(multinode=True, actor_node_ips=list(actor)),
        trainer=dict(nnodes=len(actor), n_gpus_per_node=2 // len(actor)),
        actor_rollout_ref=dict(
            rollout=dict(nnodes=len(rollout), n_gpus_per_node=2 // len(rollout), rlk_node_ips=list(rollout))
        ),
    )


@pytest.mark.parametrize(
    "actor,rollout",
    [
        (("10.0.0.1",), ("10.0.0.2",)),
        (("10.0.0.1", "10.0.0.2"), ("10.0.0.1", "10.0.0.2")),
        (("10.0.0.1", "10.0.0.2"), ("10.0.0.3",)),
    ],
)
def test_supported_placements(actor, rollout):
    assert cluster.topology(config(actor, rollout))["enabled"]


@pytest.mark.parametrize("failure", ["single", "duplicates", "gpus", "implicit", "bad_ip"])
def test_bad_layout_rejected(failure):
    c = config()
    if failure == "single":
        c = config(("10.0.0.1",), ("10.0.0.1",))
    elif failure == "duplicates":
        c["rlk"]["actor_node_ips"] = ["10.0.0.1"] * 2
    elif failure == "gpus":
        c["trainer"]["n_gpus_per_node"] = 2
    elif failure == "implicit":
        c["rlk"]["multinode"] = False
    else:
        c["rlk"]["actor_node_ips"][0] = "not-an-ip"
    with pytest.raises(ValueError):
        cluster.topology(c)


def test_old_single_node_config_still_valid():
    c = config(("10.0.0.1",), ("10.0.0.2",))
    c["rlk"] = {}
    c["actor_rollout_ref"]["rollout"].pop("rlk_node_ips")
    assert not cluster.topology(c)["enabled"]


def test_storage_requires_real_shared_visibility(tmp_path):
    (tmp_path / "challenge").write_text("nonce")
    for node in ("a", "b"):
        cluster.check_storage(tmp_path, "nonce", node)
    cluster.check_storage(tmp_path, "nonce", "a", ("a", "b"))
    (tmp_path / "b.json").unlink()
    with pytest.raises(FileNotFoundError):
        cluster.check_storage(tmp_path, "nonce", "a", ("a", "b"))
    with pytest.raises(RuntimeError):
        cluster.check_storage(tmp_path, "stale", "a")


def test_equal_pids_on_different_nodes_and_restart_detection():
    manifest = {str(i): dict(sha256=str(i), dtype="torch.float32", shape=[1]) for i in (0, 1)}
    acks = [
        dict(
            pipeline_stage=i,
            pid=100,
            worker_identity=f"node{i}:100:boot1",
            weight_version=1,
            parameter_load_completed=True,
            verified_original_tensors={str(i): manifest[str(i)]},
        )
        for i in (0, 1)
    ]
    previous = contract.validate_stage_acks(acks, manifest, 1)
    assert previous[0] != previous[1]
    acks[1]["worker_identity"] = "node1:100:boot2"
    with pytest.raises(RuntimeError):
        contract.validate_stage_acks(acks, manifest, 1, previous)
    acks[1]["worker_identity"] = acks[0]["worker_identity"]
    with pytest.raises(RuntimeError):
        contract.validate_stage_acks(acks, manifest, 1)


def test_actual_placement_detects_colocation_or_missing_rank():
    plan = cluster.topology(config())
    nodes = {"a": "10.0.0.1", "b": "10.0.0.2"}
    actors = [dict(rank=i, node_id=node, pci_bus_id="gpu0") for i, node in enumerate(nodes)]
    rollouts = [dict(rank=i, node_id=node, pci_bus_id="gpu1") for i, node in enumerate(nodes)]
    assert cluster.verify_placement(plan, actors, rollouts, nodes)
    # PCI addresses can be equal across hosts; only node+PCI must be unique.
    rollouts[1]["pci_bus_id"] = "gpu0"
    with pytest.raises(RuntimeError, match="overlaps"):
        cluster.verify_placement(plan, actors, rollouts, nodes)
    with pytest.raises(RuntimeError, match="missing"):
        cluster.verify_placement(plan, actors[:1], rollouts, nodes)


def test_pinned_pool_rolls_back_on_reservation_failure(monkeypatch):
    module = load("cluster")
    removed, specs = [], []
    base = types.ModuleType("verl.single_controller.ray.base")
    base.RayResourcePool = lambda *a, **k: types.SimpleNamespace(pgs=None)
    monkeypatch.setitem(sys.modules, "verl.single_controller.ray.base", base)
    import importlib

    import ray.util.placement_group as unused  # noqa: F401

    placement = importlib.import_module("ray.util.placement_group")

    def create(bundles, **kwargs):
        specs.append((bundles, kwargs))
        return types.SimpleNamespace(ready=lambda: "pending")

    monkeypatch.setattr(placement, "placement_group", create)
    monkeypatch.setattr(placement, "remove_placement_group", removed.append)

    def timeout(*args, **kwargs):
        raise TimeoutError("no allocation")

    monkeypatch.setattr(module.ray, "get", timeout)
    with pytest.raises(TimeoutError):
        module.pinned_pool(["10.0.0.2", "10.0.0.1"], 1, "test", 2)
    assert len(removed) == 2
    assert specs[0][0] == [{"CPU": 2, "GPU": 1, "node:10.0.0.1": 0.001}]
    assert all(s[1]["strategy"] == "STRICT_PACK" for s in specs)


def test_full_trainer_contract_keeps_numerical_gates():
    from omegaconf import OmegaConf

    config = OmegaConf.load(ROOT / "config/dsv4_grpo.yaml")
    config.rlk.sync_dir = "/shared/job/policy-sync"
    contract.validate_trainer_config(config)
    config.rlk.multinode = True
    config.rlk.actor_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.trainer.nnodes = 2
    config.trainer.n_gpus_per_node = 1
    config.actor_rollout_ref.rollout.rlk_node_ips = ["10.0.0.1", "10.0.0.2"]
    config.actor_rollout_ref.rollout.nnodes = 2
    config.actor_rollout_ref.rollout.n_gpus_per_node = 1
    contract.validate_trainer_config(config)
    config.actor_rollout_ref.actor.megatron.expert_model_parallel_size = 2
    with pytest.raises(ValueError, match="expert_model_parallel_size"):
        contract.validate_trainer_config(config)

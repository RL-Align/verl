# SPDX-License-Identifier: Apache-2.0
"""Explicit Ray placement and shared-storage preflight for PP2 across hosts."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from .cluster_contract import check_storage, topology, verify_placement


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(root, nonce, peer_ids, paths):
    node_id = ray.get_runtime_context().get_node_id()
    receipt = check_storage(root, nonce, node_id, peer_ids)
    if not peer_ids:
        from .runtime import process_identity

        receipt["source_identity"] = process_identity("cluster-preflight")
        receipt["inputs"] = {p: _sha(p) for p in paths}
    return receipt


def preflight(config):
    """Fail before allocating GPU workers if files or source revisions differ."""
    plan = topology(config)
    if not plan["enabled"]:
        return
    live = {n["NodeManagerAddress"]: n for n in ray.nodes() if n["Alive"]}
    ips = sorted(set(plan["actor_node_ips"] + plan["rollout_node_ips"]))
    for ip in ips:
        if ip not in live:
            raise RuntimeError(f"allocated node {ip} is not alive in Ray")
        required = sum(plan[f"{role}_gpus_per_node"] for role in ("actor", "rollout") if ip in plan[f"{role}_node_ips"])
        if live[ip]["Resources"].get("GPU", 0) < required:
            raise RuntimeError(f"{ip}: needs {required} allocated GPUs")
    root = Path(config.rlk.sync_dir)
    root.mkdir(parents=True, exist_ok=True)
    challenge = root / ("preflight-" + uuid4().hex)
    challenge.mkdir()
    nonce = uuid4().hex
    (challenge / "challenge").write_text(nonce)
    model = Path(config.actor_rollout_ref.model.path)
    metadata = json.loads((model / "config.json").read_text())
    split = [int(x) for x in os.environ.get("RLK_DSV4_PP_SPLIT", "3,1").split(",")]
    if metadata.get("num_hidden_layers") != sum(split):
        raise ValueError(f"checkpoint layers do not match RLK_DSV4_PP_SPLIT={split}")
    paths = [str(model / name) for name in ("config.json", "tokenizer.json")]
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        raise ValueError("checkpoint has no safetensors weights")
    paths.extend(str(p) for p in weights)
    paths.extend(str(p) for p in model.glob("*.index.json"))
    for field in ("train_files", "val_files"):
        files = config.data[field]
        paths.extend([files] if isinstance(files, str) else list(files))
    paths.extend(config.rlk.get("preflight_files", []))
    expected_inputs = {p: _sha(p) for p in paths}
    from .runtime import process_identity

    expected_source = process_identity("cluster-driver")
    ids = [live[ip]["NodeID"] for ip in ips]
    probe = ray.remote(num_cpus=0)(_probe)

    def submit(peers):
        return [
            probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n, soft=False)).remote(
                str(challenge), nonce, peers, paths
            )
            for n in ids
        ]

    receipts = ray.get(submit([]), timeout=1800)
    ray.get(submit(ids), timeout=180)
    for receipt in receipts:
        source = receipt["source_identity"]
        if receipt["inputs"] != expected_inputs:
            raise RuntimeError("model/data metadata differ across nodes")
        if source["dependencies"] != expected_source["dependencies"]:
            raise RuntimeError("dependency versions differ across nodes")
        for name, project in source["projects"].items():
            if project["dirty"] or project["sha"] != expected_source["projects"][name]["sha"]:
                raise RuntimeError(f"{name}: dirty or different code across nodes")
    (root / "cluster-preflight.json").write_text(
        json.dumps(dict(plan=plan, nodes={live[ip]["NodeID"]: ip for ip in ips}, receipts=receipts), indent=2)
    )


def pinned_pool(node_ips, gpus_per_node, name, cpus_per_gpu):
    """Reserve GPU bundles on exactly the requested Ray node IPs, with a timeout."""
    from ray.util.placement_group import placement_group, remove_placement_group

    from verl.single_controller.ray.base import RayResourcePool

    pool = RayResourcePool(
        [gpus_per_node] * len(node_ips), use_gpu=True, max_colocate_count=cpus_per_gpu, name_prefix=name
    )
    groups = []
    try:
        for ip in sorted(node_ips):
            bundles = [dict(CPU=cpus_per_gpu, GPU=1, **{f"node:{ip}": 0.001}) for _ in range(gpus_per_node)]
            groups.append(placement_group(bundles, strategy="STRICT_PACK", name=f"{name}-{ip}-{uuid4().hex}"))
        ray.get([pg.ready() for pg in groups], timeout=180)
    except Exception:
        for pg in groups:
            remove_placement_group(pg)
        raise
    pool.pgs = groups
    return pool


def actor_pool_manager(manager, node_ips, gpus_per_node):
    """Keep native resource-manager behavior and replace only the opt-in actor pool."""
    from verl.single_controller.ray.base import ResourcePoolManager

    class PinnedManager(ResourcePoolManager):
        def create_resource_pool(self):
            super().create_resource_pool()
            self.resource_pool_dict["global_pool"] = pinned_pool(node_ips, gpus_per_node, "rlk-actor", 3)

    return PinnedManager(
        resource_pool_spec=manager.resource_pool_spec,
        mapping=manager.mapping,
        max_colocate_count=manager.max_colocate_count,
    )


def record_placement(config, actors, rollout):
    """Check actual GPU ownership before any optimizer update or policy publication."""
    if not config.rlk.get("multinode", False):
        return
    root = Path(config.rlk.sync_dir)
    pre = json.loads((root / "cluster-preflight.json").read_text())
    result = verify_placement(pre["plan"], actors, rollout, pre["nodes"])
    (root / "cluster-placement.json").write_text(json.dumps(result, indent=2))

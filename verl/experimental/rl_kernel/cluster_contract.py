# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the opt-in four-layer PP2 multi-node recipe."""

import ipaddress
import json
from pathlib import Path


def topology(config):
    """Validate explicit placement without enabling new numerical layouts."""
    rlk = config.get("rlk", {})
    rollout = config["actor_rollout_ref"]["rollout"]
    trainer = config["trainer"]
    enabled = rlk.get("multinode", False)
    actor_ips = list(rlk.get("actor_node_ips", []))
    rollout_ips = list(rollout.get("rlk_node_ips", []))
    for role, settings, ips in [("actor", trainer, actor_ips), ("rollout", rollout, rollout_ips)]:
        nodes, gpus = settings["nnodes"], settings["n_gpus_per_node"]
        if type(nodes) is not int or type(gpus) is not int or (nodes, gpus) not in [(1, 2), (2, 1)]:
            raise ValueError(f"{role}: only PP2/DP1 on 1x2 or 2x1 GPUs is implemented")
        if enabled:
            if len(ips) != nodes or len(set(ips)) != nodes:
                raise ValueError(f"{role}: supply one unique Ray node IP per node")
            for ip in ips:
                ipaddress.ip_address(ip)
        elif nodes != 1 or ips:
            raise ValueError("explicit node placement requires rlk.multinode=true")
    if enabled and len(set(actor_ips + rollout_ips)) < 2:
        raise ValueError("multinode requires at least two distinct physical node IPs")
    return dict(
        enabled=enabled,
        actor_node_ips=actor_ips,
        rollout_node_ips=rollout_ips,
        actor_gpus_per_node=trainer["n_gpus_per_node"],
        rollout_gpus_per_node=rollout["n_gpus_per_node"],
    )


def worker_key(record):
    """Include node/process identity; retain compatibility with measured old evidence."""
    return record.get("worker_identity", record["pid"])


def check_storage(root, nonce, node_id, peer_ids=()):
    """Read the driver nonce and every peer marker before publishing our receipt."""
    root = Path(root)
    if (root / "challenge").read_text() != nonce:
        raise RuntimeError("shared storage challenge differs")
    for peer in peer_ids:
        if (root / f"{peer}.json").read_text() != json.dumps(dict(node_id=peer, nonce=nonce)):
            raise RuntimeError("shared storage peer receipt differs")
    path = root / f"{node_id}.json"
    if not peer_ids:
        with path.open("x") as stream:
            stream.write(json.dumps(dict(node_id=node_id, nonce=nonce)))
    return dict(node_id=node_id, nonce=nonce)


def verify_placement(plan, actor_records, rollout_records, nodes):
    """Verify two ranks per role and disjoint physical GPUs against the declared plan."""
    all_devices = set()
    actual = {}
    for role, records in [("actor", actor_records), ("rollout", rollout_records)]:
        if len(records) != 2 or {r["rank"] for r in records} != {0, 1}:
            raise RuntimeError(f"{role}: missing or duplicate PP rank")
        counts = {}
        for record in records:
            node_id = record["node_id"]
            if node_id not in nodes:
                raise RuntimeError("worker node is absent from preflight")
            ip = nodes[node_id]
            counts[ip] = counts.get(ip, 0) + 1
            device = (node_id, record["pci_bus_id"])
            if device in all_devices:
                raise RuntimeError("actor/rollout physical GPU assignment overlaps")
            all_devices.add(device)
        expected = {ip: plan[f"{role}_gpus_per_node"] for ip in plan[f"{role}_node_ips"]}
        if counts != expected:
            raise RuntimeError(f"{role}: observed placement {counts} differs from {expected}")
        actual[role] = records
    return actual

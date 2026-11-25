# SPDX-License-Identifier: Apache-2.0
"""Complete rank-local phase recovery for the same actor topology.

The original frozen quantized buffers, optimizer masters/moments, scheduler,
weight version and RNG are retained. This is process recovery, not an HF export
or cross-topology optimizer conversion. Success still requires real GPU tests.
"""

import copy
import hashlib
import io
import json
import os
import random
import struct
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = cpu_copy(item)
        return result
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def state_digest(value):
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            h.update(b"tensor" + str(data.dtype).encode() + repr(tuple(data.shape)).encode())
            h.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode() + str(len(item)).encode())
            for child in item:
                visit(child)
        elif isinstance(item, np.ndarray):
            h.update(b"ndarray" + str(item.dtype).encode() + repr(item.shape).encode() + item.tobytes())
        elif isinstance(item, io.BytesIO):
            h.update(b"bytesio" + item.getvalue())
        elif item is None or isinstance(item, (str, int, float, bool, bytes)):
            raw = item if isinstance(item, bytes) else repr(item).encode()
            h.update(type(item).__name__.encode() + struct.pack("<Q", len(raw)) + raw)
        else:
            raise TypeError(f"unsupported recovery state type: {type(item)}")

    visit(value)
    return h.hexdigest()


def capture(engine, provenance, *, capture_cuda=True):
    if engine.optimizer is None or engine.lr_scheduler is None:
        raise ValueError("phase recovery requires initialized optimizer and scheduler")
    return cpu_copy(
        dict(
            schema_version=1,
            provenance=provenance,
            weight_version=engine.rlk_weight_version,
            models=[m.state_dict() for m in engine._rlk_models],
            optimizer=engine.optimizer.state_dict(),
            scheduler=engine.lr_scheduler.state_dict(),
            optimizer_evidence=getattr(engine, "_rlk_previous_optimizer", None),
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if capture_cuda else None,
            ),
        )
    )


def restore(engine, payload, expected_provenance, *, restore_cuda=True):
    if payload["schema_version"] != 1 or payload["provenance"] != expected_provenance:
        raise ValueError("phase checkpoint source/model/topology provenance differs")
    if engine.rlk_weight_version != payload["weight_version"]:
        raise ValueError("initialize the new actor at the saved weight version before binding operators")
    if len(engine._rlk_models) != len(payload["models"]):
        raise ValueError("model chunk count changed across phases")
    for model, state in zip(engine._rlk_models, payload["models"], strict=True):
        model.load_state_dict(state, strict=True, assign=False)
        if state_digest(model.state_dict()) != state_digest(state):
            raise RuntimeError("actor model/buffer bytes changed during restore")
    engine.optimizer.load_state_dict(payload["optimizer"])
    engine.lr_scheduler.load_state_dict(payload["scheduler"])
    if state_digest(engine.optimizer.state_dict()) != state_digest(payload["optimizer"]):
        raise RuntimeError("optimizer masters/moments/scalars changed during restore")
    if state_digest(engine.lr_scheduler.state_dict()) != state_digest(payload["scheduler"]):
        raise RuntimeError("scheduler state changed during restore")
    previous_evidence = payload["optimizer_evidence"]
    engine._rlk_previous_optimizer = None
    if previous_evidence is not None:
        from examples.operators.dsv4_4l.optimizer_evidence import optimizer_snapshot

        # The old process identity and data pointers cannot survive reconstruction.
        # Value equality above proves recovery; subsequent in-process checks start
        # from the new native optimizer identity and retain a separate old receipt.
        engine._rlk_previous_optimizer = optimizer_snapshot(engine)
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    if restore_cuda:
        if payload["rng"]["cuda"] is None:
            raise ValueError("GPU phase recovery requires actual saved GPU RNG state")
        torch.cuda.set_rng_state(payload["rng"]["cuda"])
    return dict(
        weight_version=engine.rlk_weight_version,
        lifecycle="process rebuilt; original optimizer values verified, identity rebound explicitly",
        previous_process=(previous_evidence or {}).get("pid"),
        current_process=os.getpid(),
        optimizer_state_digest=state_digest(payload["optimizer"]),
        scheduler_state_digest=state_digest(payload["scheduler"]),
        model_state_digest=state_digest(payload["models"]),
    )


def save_rank(engine, directory, provenance):
    import torch.distributed as dist

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rank, world = dist.get_rank(), dist.get_world_size()
    path = root / f"rank-{rank}.pt"
    if path.exists() or (root / "manifest.json").exists():
        raise FileExistsError("never overwrite an earlier phase checkpoint")
    payload = capture(engine, provenance)
    tmp = root / f".rank-{rank}-{uuid4().hex}.pending"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    record = dict(
        rank=rank,
        file=path.name,
        sha256=digest,
        bytes=path.stat().st_size,
        state_digest=state_digest(payload),
        weight_version=engine.rlk_weight_version,
    )
    records = [None] * world
    dist.all_gather_object(records, record)
    if rank == 0:
        if {r["rank"] for r in records} != set(range(world)):
            raise RuntimeError("phase checkpoint rank coverage differs")
        manifest = dict(
            schema_version=1,
            world_size=world,
            provenance=provenance,
            weight_version=engine.rlk_weight_version,
            ranks=records,
        )
        pending = root / "manifest.pending"
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(pending, root / "manifest.json")
    dist.barrier()
    return record


def load_rank(engine, directory, expected_provenance):
    import torch.distributed as dist

    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    world, rank = dist.get_world_size(), dist.get_rank()
    if manifest["world_size"] != world or manifest["provenance"] != expected_provenance:
        raise ValueError("phase recovery topology or provenance changed")
    records = manifest["ranks"]
    if len(records) != world or {r["rank"] for r in records} != set(range(world)):
        raise ValueError("phase checkpoint missing or duplicate ranks")
    rec = next(r for r in records if r["rank"] == rank)
    path = root / rec["file"]
    if path.name != f"rank-{rank}.pt" or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("rank checkpoint file ownership differs")
    with path.open("rb") as f:
        if path.stat().st_size != rec["bytes"] or hashlib.file_digest(f, "sha256").hexdigest() != rec["sha256"]:
            raise ValueError("rank checkpoint file hash differs")
    # Only trusted files produced by save_rank in this local experiment are accepted.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if state_digest(payload) != rec["state_digest"] or payload["weight_version"] != manifest["weight_version"]:
        raise ValueError("rank payload version/state differs from committed manifest")
    result = restore(engine, payload, expected_provenance)
    dist.barrier()
    return result

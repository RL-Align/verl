# SPDX-License-Identifier: Apache-2.0
"""Independent CPU replay of actual phase score files, rank receipts and readbacks.

This accepts fixed-weight scoring only. Three-update/target-load acceptance also
requires the update, recovery and post-v3 evidence; startup never qualifies.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from rl_engine.integrations.common.response_bitwise import compare_fp32_bytes, verify_saved_pair

from verl.experimental.rl_kernel.phase_checkpoint import state_digest


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def verify(args):
    profile = read(args.profile)
    if profile["id"] not in ("N1", "N2", "N3", "N4"):
        raise ValueError("N5 requires separately qualified EP evidence and must not bypass its gate")
    target = profile["actor"]
    if (target["pp"], target["tp"], target["ep"], target["etp"], target["dp"]) != (2, 1, 1, 1, 1):
        raise ValueError("revised primary actor topology differs")
    alloc = read(args.allocation)
    expected_pci = [row["pci_bus_id"].lower() for row in alloc]
    actors = [read(args.actor / f"actor-rank-{rank}.json") for rank in range(2)]
    if [r["rank"] for r in actors] != [0, 1] or [r["original_layers"] for r in actors] != [[0, 1, 2, 3], [4, 5, 6]]:
        raise ValueError("actual seven-layer actor PP ownership differs")
    if [r["pci_bus_id"].lower() for r in actors] != expected_pci[:2]:
        raise ValueError("actual actor GPU PCI placement differs")
    rollout = read(args.rollout / "rollout-ranks.json")
    tp = profile["rollout"]["tp"]
    if len(rollout) != tp or {r["tensor_parallel_rank"] for r in rollout} != set(range(tp)):
        raise ValueError("actual rollout TP ranks are incomplete or duplicated")
    by_rank = sorted(rollout, key=lambda r: r["tensor_parallel_rank"])
    if [r["pci_bus_id"].lower() for r in by_rank] != expected_pci[:tp]:
        raise ValueError("actual rollout PCI placement differs")
    payload = torch.load(args.policy, map_location="cpu", weights_only=True)
    model = read(args.model_lock)
    eligible = {k: v for k, v in model["tensors"].items() if v["dtype"] in ("BF16", "F32", "I64")}
    if set(payload["weights"]) != set(payload["manifest"]) or set(payload["weights"]) != set(eligible):
        raise ValueError("policy does not cover every original nonquantized parameter/routing tensor")
    for name, value in payload["weights"].items():
        declared = payload["manifest"][name]
        if (
            list(value.shape) != declared["shape"]
            or str(value.dtype) != declared["dtype"]
            or tensor_sha(value) != declared["sha256"]
        ):
            raise ValueError("policy payload differs from its original-tensor manifest: " + name)
        original_dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}[eligible[name]["dtype"]]
        if value.dtype != original_dtype or list(value.shape) != eligible[name]["shape"]:
            raise ValueError("policy changed original tensor shape: " + name)
        if payload["weight_version"] == 0 and tensor_sha(value) != eligible[name]["sha256"]:
            raise ValueError("initial policy differs from original frozen snapshot: " + name)
    readbacks = read(args.rollout / "policy-raw-readback.json")
    if len(readbacks) != tp or {r["tensor_parallel_rank"] for r in readbacks} != set(range(tp)):
        raise ValueError("raw parameter readback misses TP ranks")
    partitions = {name: [] for name in payload["weights"]}
    for receipt in readbacks:
        rank = receipt["tensor_parallel_rank"]
        if receipt["weight_version"] != payload["weight_version"] or receipt["original_layer_ids"] != list(range(7)):
            raise ValueError("stale weight version or incomplete rollout layers")
        expected_name = f"pp0-tp{rank}.pt"
        if Path(receipt["raw_readback_file"]).name != expected_name:
            raise ValueError("raw readback filename does not match rank")
        path = args.rollout / "readback" / expected_name
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != receipt["raw_readback_sha256"]:
                raise ValueError("raw readback file hash differs")
        actual = torch.load(path, map_location="cpu", weights_only=True)
        if actual["weight_version"] != payload["weight_version"] or set(actual["tensors"]) != set(payload["weights"]):
            raise ValueError("raw readback tensor-name/version coverage differs")
        for name, tensor in actual["tensors"].items():
            expected = payload["weights"][name]
            ack = receipt["verified_original_tensors"][name]
            part = ack.get("tp_partition")
            if part is not None:
                if name not in ("embed.weight", "head.weight") and not name.endswith(".attn.attn_sink"):
                    raise ValueError("unqualified tensor partition in independent readback")
                if part["axis"] != 0 or part["logical_shape"] != list(expected.shape):
                    raise ValueError("raw readback partition geometry differs")
                a, b = part["start"], part["stop"]
                if not 0 <= a <= b <= expected.shape[0]:
                    raise ValueError("readback partition is out of range")
                expected = expected.narrow(0, a, b - a)
                partitions[name].append((a, b))
            else:
                partitions[name].append(None)
            if (
                tensor.dtype != expected.dtype
                or tensor.shape != expected.shape
                or not torch.equal(tensor.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
            ):
                raise ValueError("actual rollout tensor bits differ from actor: " + name)
            if tensor_sha(tensor) != ack["sha256"]:
                raise ValueError("acknowledged tensor hash differs from independent raw bytes")
    for name, ranges in partitions.items():
        if all(r is None for r in ranges):
            continue
        if any(r is None for r in ranges):
            raise ValueError("mixed replicated and sharded tensor ownership")
        ranges.sort()
        if (
            ranges[0][0] != 0
            or ranges[-1][1] != payload["weights"][name].shape[0]
            or any(a[1] != b[0] for a, b in zip(ranges[:-1], ranges[1:], strict=True))
        ):
            raise ValueError("missing or overlapping logical tensor partition")
    for path in (args.actor / "scores.json", args.rollout / "scores.json"):
        if read(path)["producer"].get("weight_version") != payload["weight_version"]:
            raise ValueError("score producer used a different weight version")
    scores = verify_saved_pair(
        args.suite, args.actor / "scores.json", args.rollout / "scores.json", args.snapshot_sha256
    )
    if scores["status"] != "PASS":
        return dict(scores, status="FAIL", rank_and_weight_readback="PASS")
    # Negative control derives from actual model scores and never replaces them.
    manifest = read(args.actor / "scores.json")
    raw = (args.actor / manifest["rows"][0]["file"]).read_bytes()
    perturbed = bytearray(raw)
    perturbed[0] ^= 1
    negative = compare_fp32_bytes(raw, perturbed, len(raw) // 4)
    if negative["status"] != "FAIL" or negative["mismatch_count"] != 1:
        raise RuntimeError("actual-score one-bit negative control failed to detect perturbation")
    (args.output.parent / "negative-control.f32").write_bytes(perturbed)
    return dict(
        scores,
        scope="fixed-weight scoring only; not GRPO or target-load acceptance",
        rank_and_weight_readback="PASS",
        actual_actor_ranks=2,
        actual_rollout_ranks=tp,
        negative_control=negative,
        policy_state_digest=state_digest(payload["weights"]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["profile", "allocation", "actor", "rollout", "policy", "model-lock", "suite", "output"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--snapshot-sha256", required=True)
    args = ap.parse_args()
    try:
        result = verify(args)
    except Exception as e:
        result = dict(status="FAIL", error=f"{type(e).__name__}: {e}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

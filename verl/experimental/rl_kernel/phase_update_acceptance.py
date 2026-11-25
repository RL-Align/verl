# SPDX-License-Identifier: Apache-2.0
"""Independently verify three real native updates and their complete recovery."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from examples.operators.dsv4_4l.optimizer_evidence import verify_optimizer_transition
from rl_engine.integrations.common.response_bitwise import compare_fp32_bytes

from verl.experimental.rl_kernel.phase_checkpoint import state_digest


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def verify(root):
    steps = read(root / "steps.json")
    if len(steps) != 3 or [s["update"] for s in steps] != [1, 2, 3]:
        raise ValueError("exactly three consecutive updates are required")
    previous_payloads = {}
    checked = []
    for step in steps:
        version = step["update"]
        update = root / f"update-{version}"
        checkpoint = root / f"checkpoint-v{version}"
        manifest = read(checkpoint / "manifest.json")
        if (
            manifest["world_size"] != 2
            or manifest["weight_version"] != version
            or {r["rank"] for r in manifest["ranks"]} != {0, 1}
        ):
            raise ValueError("complete actual PP2 checkpoint/version coverage is required")
        records = {r["rank"]: r for r in manifest["ranks"]}
        for rank in range(2):
            record = records[rank]
            file = checkpoint / record["file"]
            if file.name != f"rank-{rank}.pt":
                raise ValueError("checkpoint rank filename differs")
            with file.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError("raw recovery checkpoint hash differs")
            payload = torch.load(file, map_location="cpu", weights_only=False, mmap=True)
            if payload["weight_version"] != version or state_digest(payload) != record["state_digest"]:
                raise ValueError("raw saved state differs from rank manifest")
            native = read(update / f"actor-rank-{rank}-v{version}.json")
            verify_optimizer_transition(native["optimizer_before"], native["optimizer_after"], version=version)
            if not native["gradients_finite"] or not native["gradients_nonzero"] or not native["changed_parameters"]:
                raise ValueError("no real effective native gradient/parameter change")
            for name, sha in native["parameter_after"].items():
                if digest(payload["models"][0][name]) != sha:
                    raise ValueError("actual saved model differs from native updated parameter: " + name)
            directions = {r["direction"] for r in native["pipeline_transfers"]}
            required = {"forward_send", "backward_recv"} if rank == 0 else {"forward_recv", "backward_send"}
            if not required.issubset(directions):
                raise ValueError("actual PP forward/backward transfer is missing")
            if version > 1:
                recovered = read(update / f"restore-rank-{rank}.json")
                previous = previous_payloads[rank]
                if recovered["optimizer_state_digest"] != state_digest(previous["optimizer"]) or recovered[
                    "scheduler_state_digest"
                ] != state_digest(previous["scheduler"]):
                    raise ValueError("optimizer/scheduler history was not restored exactly")
                for name, sha in native["parameter_before"].items():
                    if digest(previous["models"][0][name]) != sha:
                        raise ValueError("actor parameters changed/reset between update phases")
            previous_payloads[rank] = payload
        batch = read(root / f"v{version - 1}-generate/generated-batch.json")
        expected = Counter(tuple(c["prompt_ids"] + c["response_ids"]) for c in batch["samples"])
        observed = Counter()
        tokens = 0
        paths = sorted(update.glob(f"update-forward-rank1-v{version}-micro*.pt"))
        if not paths:
            raise ValueError("no retained actual update-forward scores")
        for path in paths:
            raw = torch.load(path, map_location="cpu", weights_only=True)
            if raw["weight_version"] != version - 1:
                raise ValueError("update-forward version differs")
            observed.update(tuple(row) for row in raw["input_token_rows"])
            mask = raw["response_mask"].bool()
            left, right = raw["current"][mask], raw["old"][mask]
            if left.dtype != torch.float32 or right.dtype != torch.float32:
                raise TypeError("update scores must remain original FP32")
            result = compare_fp32_bytes(
                left.contiguous().numpy().tobytes(), right.contiguous().numpy().tobytes(), int(mask.sum())
            )
            if result["status"] != "PASS":
                raise ValueError("native PPO update-forward bit parity failed")
            tokens += result["compared_tokens"]
        if observed != expected or tokens != sum(len(c["response_ids"]) for c in batch["samples"]):
            raise ValueError("update forward dropped/duplicated logical samples or response tokens")
        for name in [f"v{version}-actor-audit", f"v{version}-fixed-actor"]:
            audit = read(root / name / "verification.json")
            if (
                audit["status"] != "PASS"
                or audit["expected_tokens"] != audit["compared_tokens"]
                or audit["compared_tokens"] <= 0
                or audit["mismatch_count"]
            ):
                raise ValueError("post-update independent audit is incomplete, including v3")
        checked.append(dict(update=version, expected_tokens=tokens, compared_tokens=tokens, mismatch_count=0))
    return dict(
        status="PASS",
        completed_updates=3,
        scope="independent three-update/recovery/PP/bit-evidence replay",
        updates=checked,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        result = verify(args.root)
    except Exception as e:
        result = dict(status="FAIL", error=f"{type(e).__name__}: {e}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

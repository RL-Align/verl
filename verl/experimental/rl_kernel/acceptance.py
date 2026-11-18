# SPDX-License-Identifier: Apache-2.0
"""Observe fixed-sequence and fresh-generation parity on the live trainer workers."""

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

import torch

from .cluster_contract import worker_key
from .trainer_contract import strict_response_parity


def audit_version(trainer):
    root = Path(trainer.config.rlk.sync_dir)
    version = trainer.checkpoint_manager.version
    server = trainer.llm_server_manager.server_handles[0]
    started = time.perf_counter()
    prompts = [trainer.tokenizer.encode(p, add_special_tokens=True) for p in trainer.config.rlk.audit_prompts]
    if len(prompts) != 3 or len({len(p) for p in prompts}) != 3:
        raise ValueError("audit requires three different prompt lengths")

    async def generate():
        return await asyncio.gather(
            *[
                server.generate.remote(
                    request_id=f"rlk-new-v{version}-{i}-{uuid4().hex}",
                    prompt_ids=prompt,
                    sampling_params=dict(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        logprobs=True,
                        max_tokens=int(trainer.config.rlk.audit_response_length),
                    ),
                )
                for i, prompt in enumerate(prompts)
            ]
        )

    outputs = asyncio.run(generate())
    fresh = [
        dict(id=name, prompt_ids=p, response_ids=o.token_ids)
        for name, p, o in zip("ABC", prompts, outputs, strict=True)
    ]
    if any(not c["response_ids"] for c in fresh):
        raise RuntimeError("fresh generation must have nonempty responses")
    if version == 0:
        import os

        replay = os.environ.get("RLK_REPLAY_FIXED_CASES")
        trainer._rlk_fixed_cases = json.loads(Path(replay).read_text()) if replay else fresh
        (root / "fixed-sequences.json").write_text(json.dumps(trainer._rlk_fixed_cases, indent=2))
    groups = [("fixed", trainer._rlk_fixed_cases), ("fresh", fresh)]
    result = dict(
        weight_version=version,
        global_steps=trainer.global_steps,
        expected_tokens=0,
        compared_tokens=0,
        mismatch_count=0,
        rows=[],
        fresh_sequences=fresh,
    )

    def observe(label, left, right, extra):
        # Save original FP32 words before raising, including every failed token.
        raw = dict(actor=left.detach().cpu(), rollout=right.detach().cpu(), **extra)
        filename = f"audit-v{version}-{label}.pt"
        torch.save(raw, root / filename)
        try:
            record = strict_response_parity(left, right, torch.ones_like(left, dtype=torch.bool))
        except Exception as error:
            capture_failed_audit(trainer, label, extra["case"], selected, error)
            raise
        record.update(label=label, evidence_file=filename)
        result["rows"].append(record)
        result["expected_tokens"] += record["expected_tokens"]
        result["compared_tokens"] += record["compared_tokens"]
        (root / f"audit-v{version}-progress.json").write_text(json.dumps(result, indent=2))

    def actor_score(cases, micro):
        replies = trainer.actor_rollout_wg.rlk_score_sequences(cases, version, micro)
        replies = [r for r in replies if r is not None]
        if len(replies) != 1 or replies[0]["weight_version"] != version:
            raise RuntimeError("audit requires one current final actor stage output")
        return replies[0]

    # B1 A/B/C, each repeated, real B2 AB/BA/AC/CA, each repeated.
    orders = [
        ("A",),
        ("B",),
        ("C",),
        ("A",),
        ("B",),
        ("C",),
        ("A", "B"),
        ("B", "A"),
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("A", "C"),
        ("C", "A"),
    ]
    for group, cases in groups:
        by_id = {c["id"]: c for c in cases}
        baseline = {}
        for ordinal, order in enumerate(orders):
            selected = [by_id[n] for n in order]
            label = f"{group}-{ordinal:02d}-" + "".join(order)
            actor_reply = actor_score(selected, len(selected))
            actor = actor_reply["scores"]
            rollout = __import__("ray").get(server.rlk_score_sequences.remote(selected, f"v{version}-{label}"))
            if rollout["global_steps"] != trainer.global_steps:
                raise RuntimeError("audit rollout score metadata is stale")
            for i, (name, left, values) in enumerate(zip(order, actor, rollout["scores"], strict=True)):
                right = torch.tensor(values, dtype=torch.float32)
                extra = dict(
                    case=by_id[name],
                    weight_version=version,
                    stages=rollout["stages"],
                    actual_actor_micro_batch_size=len(selected),
                    actor_forward_batches=actor_reply["forward_batches"],
                )
                observe(label + "-" + name, left, right, extra)
                if name not in baseline:
                    baseline[name] = (left.clone(), right.clone())
                else:
                    observe(label + "-" + name + "-actor-invariance", left, baseline[name][0], extra)
                    observe(label + "-" + name + "-rollout-invariance", right, baseline[name][1], extra)
                if group == "fresh" and ordinal < 3:
                    output = outputs["ABC".index(name)]
                    if output.log_probs is None or output.extra_fields.get("global_steps") != trainer.global_steps:
                        raise RuntimeError("fresh generation lacks current raw logprobs")
                    observe(
                        label + "-" + name + "-decode",
                        left,
                        torch.tensor(output.log_probs, dtype=torch.float32),
                        extra,
                    )
    result.update(
        mean_abs_diff=0.0,
        max_abs_diff=0.0,
        finite=True,
        batch_invariance=True,
        independent_audit_seconds=time.perf_counter() - started,
    )
    filename = "initial-v0.json" if version == 0 else f"post-update-v{version}.json"
    (root / filename).write_text(json.dumps(result, indent=2))
    with (root / "parity.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("rows", "fresh_sequences")},
                sort_keys=True,
            )
            + "\n"
        )
    return result


def capture_failed_audit(trainer, label, case, selected, error):
    """Observe the same live policy after failure; never turn the failed run into PASS."""
    import ray

    root = Path(trainer.config.rlk.sync_dir)
    version = trainer.checkpoint_manager.version
    server = trainer.llm_server_manager.server_handles[0]
    failure = dict(label=label, error=repr(error), weight_version=version)
    try:
        for suffix, cases in (("single", [case]), ("batch", selected)):
            trace_label = f"failure-v{version}-{label}-{suffix}"
            actors = trainer.actor_rollout_wg.rlk_score_sequences(cases, version, len(cases), trace_label)
            rollout = ray.get(server.rlk_score_sequences.remote(cases, trace_label))
            torch.save(dict(actor=actors, rollout=rollout, cases=cases), root / f"{trace_label}.pt")
        failure["diagnostics_completed"] = True
    except Exception as diagnostic_error:
        failure["diagnostics_completed"] = False
        failure["diagnostic_error"] = repr(diagnostic_error)
    (root / f"failure-v{version}.json").write_text(json.dumps(failure, indent=2))


def stress_current_version(trainer, repeats):
    """Additional real B1/B2 requests against the already audited current actor bytes.

    No optimizer update or model reload occurs here. These are extra replay
    diagnostics, separate from the required per-version acceptance matrix.
    """
    import ray

    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("stress repeats must be positive")
    root = Path(trainer.config.rlk.sync_dir)
    version = trainer.checkpoint_manager.version
    if version != 3 or trainer.global_steps != 3:
        raise ValueError("this bounded stress probe runs after the third real update")
    server = trainer.llm_server_manager.server_handles[0]
    audit = json.loads((root / "post-update-v3.json").read_text())
    sync = json.loads((root / "sync-v3.json").read_text())
    pids = {a["pipeline_stage"]: worker_key(a) for a in sync["stages"]}
    groups = dict(fixed=trainer._rlk_fixed_cases, fresh=audit["fresh_sequences"])
    baselines = {}
    for group, cases in groups.items():
        for i, case in enumerate(cases):
            filename = f"audit-v3-{group}-{i:02d}-{case['id']}-{case['id']}.pt"
            raw = torch.load(root / filename, map_location="cpu", weights_only=True)
            if raw["weight_version"] != version or raw["case"] != case:
                raise RuntimeError("stress baseline is not this exact live policy/case")
            baselines[group, case["id"]] = (filename, raw["actor"])
    output = root / "stress-v3"
    output.mkdir(exist_ok=False)
    orders = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("A",), ("B",), ("C",)]
    count = 0
    started = time.perf_counter()
    with (output / "comparisons.jsonl").open("w") as stream:
        for iteration in range(repeats):
            group = ("fixed", "fresh")[(iteration // len(orders)) % 2]
            by_id = {c["id"]: c for c in groups[group]}
            order = orders[iteration % len(orders)]
            selected = [by_id[name] for name in order]
            label = f"stress-{iteration:05d}-{group}-" + "".join(order)
            reply = ray.get(server.rlk_score_sequences.remote(selected, f"v3-{label}"))
            if (
                len(reply["stages"]) != 2
                or len(reply["scores"]) != len(selected)
                or reply["global_steps"] != 3
                or any(
                    stage["weight_version"] != 3 or worker_key(stage) != pids[stage["pipeline_stage"]]
                    for stage in reply["stages"]
                )
            ):
                raise RuntimeError("stress request used a different policy or worker")
            actors = [baselines[group, name][1] for name in order]
            scores = [torch.tensor(v, dtype=torch.float32) for v in reply["scores"]]
            filename = f"{iteration:05d}.pt"
            torch.save(
                dict(
                    iteration=iteration,
                    group=group,
                    order=order,
                    cases=selected,
                    actor=actors,
                    rollout=scores,
                    stages=reply["stages"],
                    weight_version=3,
                    baseline_files=[baselines[group, name][0] for name in order],
                ),
                output / filename,
            )
            rows = []
            for case, actor, rollout in zip(selected, actors, scores, strict=True):
                try:
                    row = strict_response_parity(actor, rollout, torch.ones_like(actor))
                except Exception as error:
                    capture_failed_audit(trainer, label, case, selected, error)
                    raise
                if row["expected_tokens"] != len(case["response_ids"]):
                    raise RuntimeError("stress comparison omitted response tokens")
                rows.append(row)
            tokens = sum(row["compared_tokens"] for row in rows)
            count += tokens
            stream.write(json.dumps(dict(iteration=iteration, file=filename, rows=rows)) + "\n")
            stream.flush()
            if iteration % 50 == 0:
                print("RLK_LIVE_V3_STRESS", iteration, repeats, count, flush=True)
    # Re-export/read back actual state, not just a version number, after stress.
    expected = json.loads((root / "sync-v3-raw.json").read_text())["manifest"]
    path = root / "policy-v3-stress-readback.pt"
    exports = trainer.actor_rollout_wg.rlk_export_policy(str(path))
    (root / "stress-v3-actor-readback.json").write_text(json.dumps(exports, indent=2))
    if len(exports) != 2 or any(e["weight_version"] != 3 or e["manifest"] != expected for e in exports):
        raise RuntimeError("actual actor parameters changed during inference stress")
    readbacks = ray.get(server.rlk_verify_current_policy.remote(str(path)))
    (root / "stress-v3-rollout-readback.json").write_text(json.dumps(readbacks, indent=2))
    covered = set()
    if len(readbacks) != 2:
        raise RuntimeError("final stress readback requires both actual PP stages")
    for ack in readbacks:
        if worker_key(ack) != pids[ack["pipeline_stage"]] or ack["weight_version"] != 3:
            raise RuntimeError("stress readback came from a different live worker/policy")
        current = ack["verified_original_tensors"]
        if covered.intersection(current):
            raise RuntimeError("stress readback has overlapping PP state")
        for name, item in current.items():
            if any(item[key] != expected[name][key] for key in ("sha256", "shape", "dtype")):
                raise RuntimeError("actual rollout state changed during inference stress")
        covered.update(current)
    if covered != set(expected):
        raise RuntimeError("final stress readback missed policy state")
    # Original policy-v3.pt is retained; this verified-identical transfer is temporary.
    path.unlink()
    summary = dict(
        status="PASS",
        weight_version=3,
        repeats=repeats,
        completed_repeats=repeats,
        expected_tokens=count,
        compared_tokens=count,
        mismatch_count=0,
        seconds=time.perf_counter() - started,
        pids=pids,
        additional_optimizer_updates=0,
        finite=True,
        mean_abs_diff=0.0,
        max_abs_diff=0.0,
        actor_state_unchanged=True,
        rollout_state_unchanged=True,
    )
    (root / "stress-v3.json").write_text(json.dumps(summary, indent=2))
    return summary

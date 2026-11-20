# SPDX-License-Identifier: Apache-2.0
"""Complete response-token coverage and original FP32 words for a native batch."""

import collections
import hashlib
import json


def response_batch_record(
    data, prompt_lengths, response_lengths, *, prompts, samples, version, metadata
):
    import torch

    left, right, mask = (
        data["old_log_probs"],
        data["rollout_log_probs"],
        data["response_mask"],
    )
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise TypeError("original FP32 actor and rollout scores are required")
    if left.shape != right.shape or left.shape != mask.shape or left.ndim != 2:
        raise ValueError("score and response-mask shapes disagree")
    expected = prompts * samples
    if (
        left.shape[0] != expected
        or len(prompt_lengths) != expected
        or len(response_lengths) != expected
    ):
        raise ValueError("missing or extra native response samples")
    if len(metadata) != expected or any(
        row["min_global_steps"] != version
        or row["max_global_steps"] != version
        or row["global_steps"] != version
        for row in metadata
    ):
        raise ValueError("native rollout batch contains stale or mixed policy versions")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("response mask must be binary")
    lengths = [int(x) for x in response_lengths]
    if any(n <= 0 for n in lengths):
        raise ValueError("empty native response")
    observed = mask.sum(-1).long().tolist()
    if observed != lengths:
        raise ValueError(
            "generated response tokens are missing from the comparison mask"
        )
    for row, n in zip(mask, lengths, strict=True):
        if not bool(row[:n].all()) or bool(row[n:].any()):
            raise ValueError("response-token positions are shifted or omitted")
    hashes = [
        hashlib.sha256(json.dumps(row[: int(n)].tolist()).encode()).hexdigest()
        for row, n in zip(data["prompts"], prompt_lengths, strict=True)
    ]
    groups = collections.Counter(hashes)
    if len(groups) != prompts or sorted(groups.values()) != [samples] * prompts:
        raise ValueError(
            "distinct prompt count or samples per prompt differs from the fixed load"
        )
    selected = mask.bool()
    a, b = left[selected].contiguous(), right[selected].contiguous()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("nonfinite original token logprobs")
    bad = a.view(torch.int32) != b.view(torch.int32)
    delta = (a.double() - b.double()).abs()
    return dict(
        weight_version=version,
        expected_samples=expected,
        actual_samples=left.shape[0],
        distinct_prompts=len(groups),
        samples_per_prompt=sorted(groups.values()),
        prompt_sha256=hashes,
        prompt_lengths=[int(n) for n in prompt_lengths],
        response_lengths=lengths,
        generated_response_tokens=sum(lengths),
        expected_tokens=sum(lengths),
        compared_tokens=a.numel(),
        mismatch_count=int(bad.sum()),
        finite=True,
        mean_abs_diff=float(delta.mean()),
        max_abs_diff=float(delta.max()),
        per_response_mismatch_count=[
            int(
                (
                    x[:n].contiguous().view(torch.int32)
                    != y[:n].contiguous().view(torch.int32)
                ).sum()
            )
            for x, y, n in zip(left, right, lengths, strict=True)
        ],
        missing_samples=0,
        omitted_tokens=0,
        coverage_complete=True,
        causal_alignment="Native actor shift and rollout response IDs; original tensor columns retained",
    )

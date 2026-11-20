# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Individual Contributor
import pytest
import torch

from verl.experimental.rl_kernel.batch_evidence import response_batch_record


def fixture():
    data = dict(
        old_log_probs=torch.tensor([[-1.0, -2.0], [-3.0, 0.0]]),
        rollout_log_probs=torch.tensor([[-1.0, -2.0], [-3.0, 0.0]]),
        response_mask=torch.tensor([[1, 1], [1, 0]]),
        prompts=torch.tensor([[0, 7], [0, 7]]),
    )
    return data, dict(
        prompt_lengths=[2, 2],
        response_lengths=[2, 1],
        prompts=1,
        samples=2,
        version=5,
        metadata=[dict(global_steps=5, min_global_steps=5, max_global_steps=5)] * 2,
    )


def test_all_effective_response_tokens_and_real_bos_zero_are_retained():
    data, kwargs = fixture()
    record = response_batch_record(data, **kwargs)
    assert record["expected_tokens"] == record["compared_tokens"] == 3
    assert record["mismatch_count"] == 0 and record["omitted_tokens"] == 0
    assert record["distinct_prompts"] == 1 and record["samples_per_prompt"] == [2]


def test_original_fp32_one_bit_failure_is_recorded_without_rounding():
    data, kwargs = fixture()
    data["rollout_log_probs"][0, 0] = torch.nextafter(data["old_log_probs"][0, 0], torch.tensor(0.0))
    record = response_batch_record(data, **kwargs)
    assert record["mismatch_count"] == 1 and record["per_response_mismatch_count"] == [
        1,
        0,
    ]


@pytest.mark.parametrize(
    "problem",
    ["missing_sample", "missing_token", "shifted_mask", "mixed_version", "wrong_dtype"],
)
def test_incomplete_or_misaligned_observations_fail_closed(problem):
    data, kwargs = fixture()
    if problem == "missing_sample":
        kwargs["samples"] = 3
    elif problem == "missing_token":
        kwargs["response_lengths"] = [2, 2]
    elif problem == "shifted_mask":
        data["response_mask"][1] = torch.tensor([0, 1])
    elif problem == "mixed_version":
        kwargs["metadata"] = [dict(global_steps=4, min_global_steps=4, max_global_steps=4)] * 2
    elif problem == "wrong_dtype":
        data["rollout_log_probs"] = data["rollout_log_probs"].bfloat16()
    with pytest.raises((ValueError, TypeError)):
        response_batch_record(data, **kwargs)

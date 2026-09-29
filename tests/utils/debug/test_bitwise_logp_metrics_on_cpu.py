import json
from types import SimpleNamespace

import pytest
import torch

from verl.utils.debug.metrics import calculate_bitwise_logp_metrics, calculate_debug_metrics
from verl.utils.skip.skip_manager import SkipManager


def test_bitwise_logp_ignores_padding_and_detects_signed_zero():
    data = SimpleNamespace(
        batch={
            "rollout_log_probs": torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32),
            "old_log_probs": torch.tensor([[-0.0, 2.0, -999.0]], dtype=torch.float32),
            "response_mask": torch.tensor([[1, 1, 0]]),
        }
    )
    metrics = calculate_bitwise_logp_metrics(data)
    assert metrics["training/rollout_logp_compared"] == 2
    assert metrics["training/rollout_logp_mismatch_count"] == 2
    assert metrics["training/rollout_logp_max_abs_diff"] == 1.0


def test_bitwise_logp_rejects_missing_active_values():
    data = SimpleNamespace(
        batch={
            "rollout_log_probs": torch.tensor([[float("nan")]]),
            "old_log_probs": torch.zeros(1, 1),
            "response_mask": torch.ones(1, 1),
        }
    )
    with pytest.raises(ValueError, match="finite"):
        calculate_bitwise_logp_metrics(data)


def test_debug_metrics_audit_records_step_and_bitwise_values(monkeypatch, tmp_path):
    data = SimpleNamespace(batch={
        "rollout_log_probs": torch.tensor([[0.0, -0.5, 10.0]]),
        "old_log_probs": torch.tensor([[-0.0, -0.5, 20.0]]),
        "response_mask": torch.tensor([[1, 1, 0]]),
        "responses": torch.tensor([[1, 2, 3]]),
    })
    monkeypatch.setenv("VERL_LOGP_BITWISE_AUDIT", "1")
    monkeypatch.setenv("VERL_LOGP_AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(SkipManager, "step", 2)

    metrics = calculate_debug_metrics(data)
    assert metrics["training/rollout_logp_mismatch_count"] == 1
    assert metrics["training/rollout_logp_compared"] == 2
    rows = [json.loads(line) for line in (tmp_path / "mismatch.jsonl").read_text().splitlines()]
    assert rows == [{"step": 2, "training/rollout_logp_compared": 2,
                     "training/rollout_logp_mismatch_count": 1,
                     "training/rollout_logp_max_abs_diff": 0.0}]
    saved = torch.load(tmp_path / "logp_step2.pt", weights_only=True)
    assert saved["rollout_log_probs"].view(torch.int32)[0, 0] != saved["old_log_probs"].view(torch.int32)[0, 0]


def test_debug_metrics_without_audit_does_not_save(monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_LOGP_BITWISE_AUDIT", raising=False)
    monkeypatch.setenv("VERL_LOGP_AUDIT_DIR", str(tmp_path))
    data = SimpleNamespace(batch={
        "rollout_log_probs": torch.tensor([[0.0, -0.5, 10.0]]),
        "old_log_probs": torch.tensor([[0.0, -0.5, 20.0]]),
        "response_mask": torch.tensor([[1, 1, 0]]),
        "responses": torch.tensor([[1, 2, 3]]),
    })
    assert "training/rollout_logp_compared" not in calculate_debug_metrics(data)
    assert not list(tmp_path.iterdir())

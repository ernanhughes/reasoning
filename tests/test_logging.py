import os
import json
import pytest
from datetime import datetime, timezone

from reasoning.logs.prompt_log import PromptLog
from reasoning.logs.feature_log import FeatureLog
from reasoning.logs.prompt_result_log import PromptResultLog

LOG_DIR = "logs"
AUDIT_FILE = os.path.join(LOG_DIR, "audit_log.jsonl")

@pytest.fixture(autouse=True)
def clean_logs():
    # Clean up logs before each test
    if os.path.exists(LOG_DIR):
        for file in os.listdir(LOG_DIR):
            if file.endswith(".jsonl"):
                os.remove(os.path.join(LOG_DIR, file))

def assert_log_saved(path: str, expected_key: str):
    assert os.path.exists(path)
    with open(path) as f:
        line = json.loads(f.readline())
        assert expected_key in line
        assert "created_at" in line

def test_prompt_log():
    log = PromptLog(
        model="TinyLlama",
        dataset="test dataset",
        text="Why did the market fall?",
        tokens=[101, 202, 303],
        created_at=None  # test auto-fill
    )
    log.save()
    assert_log_saved("logs/prompt_log.jsonl", "id")

def test_feature_log():
    log = FeatureLog(
        prompt_id="test_prompt_id",
        feature_id=1090,
        reason_score=0.85,
        topk_score=0.9,
        sae_config="configs/sae/small_model.yaml",
        layer_index=12
    )
    log.save()
    assert_log_saved("logs/feature_log.jsonl", "id")

def test_prompt_result_log():
    log = PromptResultLog(
        prompt_id="prompt_1000",
        result_type="actual",
        model_recommendation=1,
        time_horizon_days=30,
        actual_price_change=0.12,
        is_correct=True,
        notes="Test case",
        created_at=None
    )
    log.save()
    assert_log_saved("logs/prompt_result_log.jsonl", "id")

def test_audit_log_created():
    # Run a log to generate audit entry
    log = PromptLog(model="Test", dataset="dd1", text="X", tokens=[1])
    log.save()

    assert os.path.exists(AUDIT_FILE)
    with open(AUDIT_FILE) as f:
        line = json.loads(f.readline())
        assert line["log_type"] == "PromptLog"
        assert "log_id" in line
        assert "timestamp" in line

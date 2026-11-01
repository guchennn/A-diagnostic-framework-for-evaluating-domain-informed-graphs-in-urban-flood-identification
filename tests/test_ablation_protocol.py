from pathlib import Path

import pytest
import yaml

from floodgnn_ceus import rq2_scarcity, train

ROOT = Path(__file__).resolve().parents[1]


def test_released_full_constants_match_original_digest():
    freeze = yaml.safe_load((ROOT / "configs/rq2_scarcity_freeze_v1_2.yaml").read_text())
    table, _ = rq2_scarcity._load_full_side(ROOT, freeze)
    constants = rq2_scarcity._full_constants(table)
    assert set(constants["D_full"]) == {"XGB", "G0", "G1"}
    assert all(len(v) == 3 for v in constants["D_full"].values())


def test_fresh_ablation_requires_local_gate_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "prepare_plan", lambda root: {})

    def reject(root, plan):
        raise ValueError("local resource probe required")

    monkeypatch.setattr(train, "require_revision_environment", reject)
    output = tmp_path / "ablation"
    with pytest.raises(ValueError, match="local resource probe"):
        rq2_scarcity.run_rq2_scarcity(tmp_path, output)
    assert not output.exists()

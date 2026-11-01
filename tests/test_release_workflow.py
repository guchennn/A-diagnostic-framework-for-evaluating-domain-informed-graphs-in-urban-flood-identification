import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from floodgnn_ceus import cli, e1
from floodgnn_ceus import reviewer_revision as revision

ROOT = Path(__file__).resolve().parents[1]
CITY_INPUTS = (
    "final_scorable_grid.parquet",
    "features_18.parquet",
    "evaluation_design/outer_partitions.parquet",
    "evaluation_design/inner_blocks.parquet",
    "evaluation_design/reveals.parquet",
    "evaluation_design/u_samples.parquet",
    "evaluation_design/evaluation_banks.parquet",
    "graphs/base_edges.parquet",
    "graphs/outer_edges.parquet",
)


@pytest.fixture
def local_root(tmp_path):
    for name in (revision.CONFIG, "configs/model_grid.yaml", "pyproject.toml", "uv.lock"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    for name in ("artifacts/protocol_lock.json", "src/floodgnn_ceus/example.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for city in ("zhengzhou", "beijing"):
        for name in CITY_INPUTS:
            path = tmp_path / "data/interim" / city / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic input")
    return tmp_path


def test_local_identity_matches_actual_city_reads(local_root, monkeypatch):
    consumed = []

    def record(path):
        consumed.append(path.relative_to(local_root).as_posix())

    monkeypatch.setattr(e1.pd, "read_parquet", record)
    monkeypatch.setattr(e1.gpd, "read_parquet", record)
    for city in ("zhengzhou", "beijing"):
        e1._read_city(local_root, city)
    files = revision._input_identity(local_root)
    paths = {item["path"] for item in files}
    assert {p for p in paths if p.startswith("data/")} == set(consumed)
    assert "artifacts/protocol_lock.json" in paths
    assert set(revision._source_identity(local_root)) <= paths
    assert all(not Path(p).is_absolute() for p in paths)
    assert not any("sensitivity" in p or "autodl" in p or "e3_task" in p for p in paths)


def test_local_lock_is_immutable_and_detects_changed_inputs(local_root, monkeypatch):
    monkeypatch.setattr(revision, "_environment", lambda: {"synthetic": "CUDA"})
    plan = revision.prepare_plan(local_root)
    revision.freeze_local_environment(local_root)
    lock_path = revision._run_root(local_root, plan) / "local_environment_lock.json"
    original = lock_path.read_bytes()
    modified = lock_path.stat().st_mtime_ns
    revision.freeze_local_environment(local_root)
    assert lock_path.stat().st_mtime_ns == modified
    revision._require_local_lock(local_root, plan)
    (local_root / "data/interim/beijing/features_18.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs differ"):
        revision._require_local_lock(local_root, plan)
    with pytest.raises(ValueError, match="do not replace"):
        revision.freeze_local_environment(local_root)
    assert lock_path.read_bytes() == original
    assert (local_root / "artifacts/protocol_lock.json").read_text() == "{}"


def test_fresh_lock_requires_bound_performance_blind_probe(local_root, monkeypatch):
    monkeypatch.setattr(revision, "_environment", lambda: {"synthetic": "CUDA"})
    plan = revision.prepare_plan(local_root)
    assert revision.preflight(local_root)["status"] == "blocked"
    lock = revision.freeze_local_environment(local_root)
    assert revision.preflight(local_root)["status"] == "blocked"
    run_root = revision._run_root(local_root, plan)
    receipt = {
        "status": "success",
        "epochs": 1,
        "cell": revision.PROBE_CELL,
        "performance_blind": True,
        "predictions_written": False,
        "metrics_computed": False,
        "source_sha256": plan["source_sha256"],
        "input_sha256": lock["input_sha256"],
        "lock_sha256": revision.sha256_file(run_root / "local_environment_lock.json"),
    }
    e1._write_json(run_root / "resource_probe.json", receipt)
    assert revision.preflight(local_root)["status"] == "ready"
    for key, value in [
        ("metrics_computed", True),
        ("input_sha256", "stale"),
        ("performance_blind", False),
        ("predictions_written", True),
    ]:
        e1._write_json(run_root / "resource_probe.json", {**receipt, key: value})
        assert revision.preflight(local_root)["status"] == "blocked"
    e1._write_json(run_root / "resource_probe.json", receipt)
    monkeypatch.setattr(revision, "_environment", lambda: {"synthetic": "changed GPU"})
    assert revision.preflight(local_root)["status"] == "blocked"


def test_cli_exposes_only_preparation_commands():
    assert cli.COMMANDS == ("fetch", "prepare", "audit-e0", "lock-protocol")
    for command in cli.COMMANDS:
        assert cli.build_parser().parse_args([command]).command == command
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run-e3"])


def test_worldpop_fetch_accepts_three_return_values(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_protocol", lambda _: object())
    monkeypatch.setattr(
        cli,
        "fetch_worldpop",
        lambda *_: (
            {"2020": SimpleNamespace(bytes=17)},
            {},
            tmp_path / "artifacts/manifests/worldpop_asset_manifest.parquet",
        ),
    )
    assert cli.main(["fetch", "--source", "worldpop_r2024b", "--root", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["file_count"] == 1
    assert result["total_bytes"] == 17
    assert result["manifest"].endswith("raw_source_manifest.parquet")
    assert result["source_manifest"].endswith("worldpop_asset_manifest.parquet")


def test_post_dsm_design_uses_existing_inputs_only(tmp_path, monkeypatch):
    for city in ("zhengzhou", "beijing"):
        folder = tmp_path / "data/interim" / city
        folder.mkdir(parents=True)
        for name in ("dsm_200m_float64.tif", "final_scorable_grid.parquet"):
            (folder / name).touch()
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("post-DSM preparation must not fetch or rebuild labels")

    monkeypatch.setattr(cli, "prepare_labels", forbidden)
    monkeypatch.setattr(cli, "prepare_worldcover", forbidden)
    for name in (
        "prepare_terrain",
        "prepare_features",
        "prepare_evaluation_design",
        "prepare_graphs",
    ):
        monkeypatch.setattr(cli, name, lambda root, name=name: calls.append(name))
    assert cli.main(["prepare", "--through", "design", "--root", str(tmp_path)]) == 0
    assert calls == [
        "prepare_terrain",
        "prepare_features",
        "prepare_evaluation_design",
        "prepare_graphs",
    ]


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/protocol_lock.json",
        "data/interim/zhengzhou/evaluation_design/reveals.parquet",
        "data/interim/beijing/graphs/outer_edges.parquet",
    ],
)
def test_missing_consumed_input_blocks_freeze(local_root, monkeypatch, relative):
    monkeypatch.setattr(revision, "_environment", lambda: {"synthetic": "CUDA"})
    (local_root / relative).unlink()
    with pytest.raises(FileNotFoundError):
        revision.freeze_local_environment(local_root)


def test_input_identity_is_portable(local_root):
    import shutil

    # Copy to a sibling to avoid recursively copying the destination into itself.
    target = local_root.with_name(local_root.name + "-relocated")
    shutil.copytree(local_root, target)
    assert revision._input_identity(target) == revision._input_identity(local_root)
    assert revision.prepare_plan(target) == revision.prepare_plan(local_root)


def test_preflight_cli_fails_closed_and_resolves_output_under_root(local_root, monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reviewer_revision",
            "preflight",
            "--root",
            str(local_root),
            "--output",
            "reproduced/preflight.json",
        ],
    )
    with pytest.raises(SystemExit) as error:
        revision.main()
    assert error.value.code == 1
    result = json.loads((local_root / "reproduced/preflight.json").read_text())
    assert result["status"] == "blocked"


@pytest.mark.parametrize("stage", ["osm", "worldpop", "imerg", "dsm"])
def test_source_prepare_stages_are_independent(stage, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "prepare_labels", lambda root: calls.append("labels"))
    monkeypatch.setattr(cli, "prepare_worldcover", lambda *args: calls.append("worldcover"))
    monkeypatch.setattr(cli.earthaccess, "login", lambda **kwargs: object())
    for name in ("osm", "worldpop", "imerg", "dsm"):
        monkeypatch.setattr(cli, "prepare_" + name, lambda *args, name=name: calls.append(name))
    assert cli.main(["prepare", "--through", stage, "--root", str(tmp_path)]) == 0
    assert calls == ["labels", "worldcover", stage]


def test_release_has_no_remote_operations_module():
    import ast

    assert not (ROOT / "src/floodgnn_ceus/remote.py").exists()
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "floodgnn_ceus.remote"
            elif isinstance(node, ast.Import):
                assert all(alias.name != "floodgnn_ceus.remote" for alias in node.names)
    from floodgnn_ceus import training

    assert not hasattr(e1, "run_e1_inner_batch")
    assert not hasattr(e1, "run_e1_outer_batch")
    assert not hasattr(training, "probe_training_resources")
    assert callable(revision.run_local_probe)

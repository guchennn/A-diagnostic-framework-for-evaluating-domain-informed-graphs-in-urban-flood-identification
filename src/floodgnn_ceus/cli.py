"""Source preparation and immutable protocol locking for the released workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import earthaccess
import geopandas as gpd
import requests
from earthaccess.exceptions import LoginAttemptFailure

from floodgnn_ceus import __version__
from floodgnn_ceus.audit import (
    extract_crowdflood_events,
    inspect_crowdflood_archive,
    write_crowdflood_audit,
)
from floodgnn_ceus.e0 import audit_e0, lock_protocol
from floodgnn_ceus.evaluation_design import prepare_evaluation_design
from floodgnn_ceus.features import prepare_features, prepare_worldcover
from floodgnn_ceus.graphs import prepare_graphs
from floodgnn_ceus.imerg import download_imerg, prepare_imerg, prepare_imerg_inventory
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.labels import (
    build_label_ledgers,
    label_qc_record,
    read_crowdflood_rows,
    write_label_ledgers,
    write_label_qc_report,
)
from floodgnn_ceus.osm import fetch_osm, prepare_osm
from floodgnn_ceus.protocol import load_protocol
from floodgnn_ceus.sources import (
    DownloadError,
    build_raw_source_manifest,
    fetch_public_sources,
    write_public_source_report,
)
from floodgnn_ceus.spatial import (
    admin_aoi_record,
    build_admin_aoi,
    build_candidate_grid,
    candidate_grid_record,
    write_admin_aoi,
    write_admin_aoi_report,
    write_candidate_grid,
    write_candidate_grid_report,
)
from floodgnn_ceus.terrain import prepare_dsm, prepare_terrain
from floodgnn_ceus.worldpop import fetch_worldpop, prepare_worldpop

COMMANDS = (
    "fetch",
    "prepare",
    "audit-e0",
    "lock-protocol",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(prog="floodgnn")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="fetch frozen Phase 1 sources")
    fetch.add_argument("--city", choices=("all",), default="all")
    fetch.add_argument(
        "--source",
        choices=(
            "all",
            "crowdflood_v1",
            "geoboundaries_chn_adm2",
            "worldcover_2021_v200",
            "ohsome_v1",
            "imerg_v07b",
            "worldpop_r2024b",
        ),
        default="all",
    )
    fetch.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    prepare = subparsers.add_parser("prepare", help="prepare fixed Phase 1 spatial labels")
    prepare.add_argument("--city", choices=("all",), default="all")
    prepare.add_argument(
        "--through",
        choices=(
            "labels",
            "worldcover",
            "osm",
            "worldpop",
            "imerg",
            "dsm",
            "terrain",
            "features",
            "design",
        ),
        default="labels",
    )
    prepare.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    audit = subparsers.add_parser("audit-e0", help="audit frozen E0 artifacts")
    audit.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    lock = subparsers.add_parser("lock-protocol", help="create the immutable protocol lock")
    lock.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    return parser


def _require_post_dsm_inputs(root: Path) -> None:
    missing = []
    for city_id in ("zhengzhou", "beijing"):
        city_root = root / "data" / "interim" / city_id
        for name in ("dsm_200m_float64.tif", "final_scorable_grid.parquet"):
            path = city_root / name
            if not path.is_file():
                missing.append(path)
    if missing:
        paths = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"post-DSM prerequisites are missing:\n{paths}\nrun --through dsm first"
        )


def _prepare_post_dsm(root: Path, through: str) -> dict[str, object]:
    _require_post_dsm_inputs(root)
    result = {"terrain": prepare_terrain(root), "through": "terrain"}
    if through in {"features", "design"}:
        result.update(features=prepare_features(root), through="features")
    if through == "design":
        result.update(
            evaluation_design=prepare_evaluation_design(root),
            graphs=prepare_graphs(root),
            through="design",
        )
    return result


def prepare_labels(root: Path) -> dict[str, object]:
    """Prepare the fixed CrowdFlood audit, administrative grids, and label ledgers."""
    protocol = load_protocol(root / "configs")
    crowd_source = protocol.sources["crowdflood_v1"]
    if crowd_source["privacy_status"] == "prohibited":
        raise PermissionError("CrowdFlood processing is prohibited")
    city_ids = ("zhengzhou", "beijing")
    city_events = {city_id: protocol.cities[city_id]["city_event"] for city_id in city_ids}
    crowd_root = root / "data" / "raw" / "crowdflood_v1" / str(crowd_source["record_id"])
    archive = crowd_root / crowd_source["filename"]
    crowd_audit = inspect_crowdflood_archive(archive, city_events)
    event_paths = extract_crowdflood_events(archive, crowd_root / "events", city_events)
    crowd_report = write_crowdflood_audit(
        root / "artifacts" / "reports" / "crowdflood_audit.json",
        crowd_audit,
        record_id=crowd_source["record_id"],
        record_doi=crowd_source["record_doi"],
        license_id=crowd_source["license"],
        privacy_status=crowd_source["privacy_status"],
    )

    boundary_source = protocol.sources["geoboundaries_chn_adm2"]
    boundary_path = (
        root
        / "data"
        / "raw"
        / "geoboundaries_chn_adm2"
        / boundary_source["commit"][:8]
        / "geoBoundaries-CHN-ADM2.geojson"
    )
    boundaries = gpd.read_file(boundary_path)
    aoi_records: list[dict[str, object]] = []
    grid_records: list[dict[str, object]] = []
    label_records: list[dict[str, object]] = []
    city_summary: dict[str, dict[str, object]] = {}
    resolution_m = protocol.protocol["grid"]["resolution_m"]
    for city_id, event_path in zip(city_ids, event_paths, strict=True):
        city_root = root / "data" / "interim" / city_id
        aoi = build_admin_aoi(boundaries, protocol.cities[city_id])
        aoi_path = write_admin_aoi(aoi, city_root / "admin_aoi.parquet")
        aoi_records.append(admin_aoi_record(aoi, aoi_path))

        grid = build_candidate_grid(aoi, resolution_m=resolution_m)
        grid_path = write_candidate_grid(grid, city_root / "candidate_grid.parquet")
        grid_records.append(candidate_grid_record(grid, aoi, grid_path, resolution_m=resolution_m))

        rows = read_crowdflood_rows(event_path)
        qc, labels = build_label_ledgers(
            rows,
            city_id=city_id,
            city_event=protocol.cities[city_id]["city_event"],
            source_sha256=sha256_file(event_path),
            aoi=aoi,
            grid=grid,
            resolution_m=resolution_m,
        )
        qc_path, label_path = write_label_ledgers(
            qc,
            labels,
            city_root / "label_qc_ledger.parquet",
            city_root / "label_ledger.parquet",
        )
        label_record = label_qc_record(qc, labels, event_path, qc_path, label_path)
        label_records.append(label_record)
        city_summary[city_id] = {
            "candidate_nodes": len(grid),
            "label_cells": label_record["label_cell_counts"],
            "raw_rows": len(qc),
        }

    aoi_report = write_admin_aoi_report(
        root / "artifacts" / "reports" / "admin_aoi.json",
        aoi_records,
        source_commit=boundary_source["commit"],
    )
    grid_report = write_candidate_grid_report(
        root / "artifacts" / "reports" / "candidate_grid.json",
        grid_records,
    )
    label_report = write_label_qc_report(
        root / "artifacts" / "reports" / "label_qc.json",
        label_records,
    )
    return {
        "cities": city_summary,
        "reports": [str(crowd_report), str(aoi_report), str(grid_report), str(label_report)],
        "through": "labels",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Phase 1 command."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "fetch":
        root = arguments.root.resolve()
        try:
            protocol = load_protocol(root / "configs")
            with requests.Session() as session:
                if arguments.source == "worldpop_r2024b":
                    by_year, _, source_manifest = fetch_worldpop(root, session)
                    manifest = root / "artifacts/manifests/raw_source_manifest.parquet"
                    records = list(by_year.values())
                    report = None
                elif arguments.source == "ohsome_v1":
                    records, _, manifest = fetch_osm(root, session)
                    source_manifest = None
                    report = None
                elif arguments.source == "imerg_v07b":
                    granules, _, report = prepare_imerg_inventory(root)
                    auth = earthaccess.login(strategy="interactive", persist=False)
                    records, source_manifest, manifest = download_imerg(root, granules, auth)
                else:
                    records = fetch_public_sources(
                        session,
                        root / "data" / "raw",
                        protocol,
                        arguments.source,
                    )
                    manifest = build_raw_source_manifest(
                        root / "data" / "raw",
                        root / "artifacts",
                        protocol,
                    )
                    source_manifest = None
                    report = write_public_source_report(records, root / "artifacts")
        except (
            DownloadError,
            EOFError,
            LoginAttemptFailure,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            print(f"floodgnn fetch: {error}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "file_count": len(records),
                    "manifest": str(manifest),
                    "report": str(report) if report else None,
                    "source_manifest": str(source_manifest) if source_manifest else None,
                    "total_bytes": sum(record.bytes for record in records),
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "prepare":
        try:
            root = arguments.root.resolve()
            if arguments.through in {"terrain", "features", "design"}:
                result = _prepare_post_dsm(root, arguments.through)
            else:
                result = prepare_labels(root)
            if arguments.through in {"worldcover", "osm", "worldpop", "imerg", "dsm"}:
                with requests.Session() as session:
                    worldcover = prepare_worldcover(root, session)
                result = {"labels": result, "through": "worldcover", "worldcover": worldcover}
            if arguments.through == "osm":
                with requests.Session() as session:
                    osm = prepare_osm(root, session)
                result = {**result, "through": "osm", "osm": osm}
            if arguments.through == "worldpop":
                with requests.Session() as session:
                    worldpop = prepare_worldpop(root, session)
                result = {**result, "through": "worldpop", "worldpop": worldpop}
            if arguments.through == "imerg":
                auth = earthaccess.login(strategy="interactive", persist=False)
                imerg = prepare_imerg(root, auth)
                result = {**result, "through": "imerg", "imerg": imerg}
            if arguments.through == "dsm":
                dsm = prepare_dsm(root)
                result = {**result, "through": "dsm", "dsm": dsm}
        except (
            DownloadError,
            EOFError,
            LoginAttemptFailure,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            print(f"floodgnn prepare: {error}", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    try:
        root = arguments.root.resolve()
        if arguments.command == "audit-e0":
            audit = audit_e0(root)
            result = {
                "bundle_sha256": audit["bundle_sha256"],
                "feasibility": audit["feasibility"],
                "report": str(root / "artifacts" / "reports" / "e0_audit.json"),
                "status": audit["status"],
            }
        elif arguments.command == "lock-protocol":
            result = lock_protocol(root)
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"floodgnn {arguments.command}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import os
from pathlib import Path
import json

import pytest

from geogfm.utils.hls_downloads import (
    _search_and_select_items,
    _build_aoi_from_center,
    download_hls_samples,
    main as hls_main,
)


FIXTURES = Path(__file__).parent / "fixtures"


def load_expectations(name: str) -> dict:
    with open(FIXTURES / name, "r") as f:
        return json.load(f)


@pytest.mark.integration
def test_select_items_santa_barbara_2018_s30(tmp_path: Path):
    exp = load_expectations("hls_expectations_sb_2018_s30.json")

    aoi = _build_aoi_from_center(34.4138, -119.8489, buffer_km=5)
    items = _search_and_select_items(
        stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="hls2-s30",
        aoi_geometry=aoi,
        date_start="2018-01-01",
        date_end="2018-12-31",
        max_cloud=20,
        preferred_months=tuple(range(4, 11)),
        n_samples=exp["n_samples"],
        min_day_gap=15,
    )

    assert len(items) == exp["n_samples"]
    for it in items:
        assert isinstance(it.get("id"), str)
        assert it["id"].startswith(exp["id_prefix"])  # HLS.S30.
        props = it.get("properties", {})
        assert props.get("datetime", "").startswith("2018-")
        tile = props.get("hls:tile_id") or props.get("mgrs:tile") or ""
        assert tile.startswith(exp["tile_prefix"])  # T11S...


@pytest.mark.integration
def test_cli_dry_run_and_items_json(tmp_path: Path, monkeypatch):
    exp = load_expectations("hls_expectations_sb_2018_s30.json")
    out_dir = tmp_path / "out"
    items_json = tmp_path / "items.json"

    argv = [
        "--start",
        "2018-01-01",
        "--end",
        "2018-12-31",
        "--collection",
        "hls2-s30",
        "--center-lat",
        "34.4138",
        "--center-lon",
        "-119.8489",
        "--buffer-km",
        "5",
        "--n-samples",
        str(exp["n_samples"]),
        "--max-cloud",
        "20",
        "--out-dir",
        str(out_dir),
        "--dry-run",
        "--print-config",
        "--print-items",
        "--items-json-out",
        str(items_json),
    ]

    rc = hls_main(argv)
    assert rc == 0

    assert items_json.exists()
    data = json.loads(items_json.read_text())
    assert len(data) == exp["n_samples"]
    for d in data:
        assert d["id"].startswith(exp["id_prefix"])  # HLS.S30.
        tile = d.get("hls:tile_id") or d.get("mgrs:tile") or ""
        assert tile.startswith(exp["tile_prefix"])  # T11S...


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_LIVE_DOWNLOADS", "0") != "1", reason="Set RUN_LIVE_DOWNLOADS=1 to run live download test")
def test_live_download_one_file(tmp_path: Path):
    # Live test that fetches and writes a single file (may take a few minutes)
    out_dir = tmp_path / "live_out"
    argv = [
        "--start",
        "2018-06-01",
        "--end",
        "2018-06-30",
        "--collection",
        "hls2-s30",
        "--center-lat",
        "34.4138",
        "--center-lon",
        "-119.8489",
        "--buffer-km",
        "3",
        "--n-samples",
        "1",
        "--max-cloud",
        "20",
        "--out-dir",
        str(out_dir),
    ]
    rc = hls_main(argv)
    assert rc == 0
    outs = sorted(out_dir.glob("*.tif"))
    assert len(outs) == 1


@pytest.mark.integration
def test_known_item_exact_id_and_dry_run_naming(tmp_path: Path):
    known = load_expectations("hls_known_item_sb_2018_feb12.json")
    aoi = _build_aoi_from_center(34.4138, -119.8489, buffer_km=5)

    # Exact-item selection by constraining date range
    items = _search_and_select_items(
        stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="hls2-s30",
        aoi_geometry=aoi,
        date_start="2018-02-12",
        date_end="2018-02-13",
        max_cloud=40,
        preferred_months=None,
        n_samples=1,
        min_day_gap=1,
    )
    assert len(items) == 1
    assert items[0]["id"] == known["id"]

    # Dry-run should compute output filename without writing
    out_dir = tmp_path / "plan"
    written = download_hls_samples(
        out_dir=out_dir,
        aoi_geometry=aoi,
        date_start="2018-02-12",
        date_end="2018-02-13",
        collection="hls2-s30",
        stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        n_samples=1,
        max_cloud=40,
        preferred_months=None,
        bands=("B02","B03","B04"),
        resolution=30,
        name_prefix="SantaBarbara",
        verbose=False,
        min_day_gap=1,
        dry_run=True,
    )
    assert len(written) == 1
    assert written[0].name == "SantaBarbara_HLS.S30.T11SLT.2018043T183031.v2.0_cropped.tif"


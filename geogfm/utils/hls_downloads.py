"""CLI and helper utilities to search and download HLS v2 scenes from MPC.

Highlights:
- MPC collections: 'hls2-s30' (Sentinel-2) and 'hls2-l30' (Landsat)
- AOI via GeoJSON polygon file OR center lat/lon + buffer_km
- Non-interactive arguments and an optional interactive mode
- Rich, colorized console output with clear, context-specific guidance
- Dry-run planning, config preview, and examples to support stand-alone use
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
from shapely.geometry import Point, shape
import rioxarray as rxr  # noqa: F401
from pystac_client import Client
import planetary_computer as pc
from stackstac import stack

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich import box
except Exception:  # pragma: no cover - rich is expected but fail gracefully
    Console = None  # type: ignore
    Table = None  # type: ignore
    Panel = None  # type: ignore
    Text = None  # type: ignore
    Prompt = None  # type: ignore
    Confirm = None  # type: ignore
    Progress = None  # type: ignore
    SpinnerColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    Markdown = None  # type: ignore
    box = None  # type: ignore


def _resolve_collection_id(user_value: str) -> str:
    v = (user_value or "").strip().lower()
    if v in {"hlss30", "hls2-s30"}:
        return "hls2-s30"
    if v in {"hlsl30", "hls2-l30"}:
        return "hls2-l30"
    # Default to Sentinel-2 flavor if ambiguous
    return "hls2-s30"


def _build_aoi_from_center(center_lat: float, center_lon: float, buffer_km: float):
    wgs84 = "EPSG:4326"
    gdf = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs=wgs84)
    gdf_m = gdf.to_crs("EPSG:3310")
    return gdf_m.buffer(buffer_km * 1000).to_crs(wgs84).iloc[0]


def _load_aoi_from_geojson(path: Path):
    data = json.loads(Path(path).read_text())
    if "type" in data and data["type"].lower() == "featurecollection":
        feats = data.get("features", [])
        if not feats:
            raise ValueError("GeoJSON FeatureCollection contains no features")
        geom = feats[0]["geometry"]
    elif "type" in data and data["type"].lower() in {"feature", "polygon", "multipolygon"}:
        geom = data.get("geometry", data)
    else:
        raise ValueError("Unrecognized GeoJSON structure for AOI")
    return shape(geom)


def _search_and_select_items(
    *,
    stac_url: str,
    collection: str,
    aoi_geometry,
    date_start: str,
    date_end: str,
    max_cloud: int,
    preferred_months: Optional[Sequence[int]],
    n_samples: int,
    min_day_gap: int,
) -> List[dict]:
    """Search MPC STAC and select items according to filters.

    Returns signed item dicts suitable for stackstac and for metadata printing.
    """
    catalog = Client.open(stac_url)
    coll_id = _resolve_collection_id(collection)
    # Microsoft Planetary Computer typically exposes HLS under a single 'hls' collection
    # with 'hls:product' distinguishing HLSS30 vs HLSL30. Fall back to that canonical form.
    if coll_id in {"hls2-s30", "hls2-l30"}:
        hls_product = "HLSS30" if coll_id.endswith("s30") else "HLSL30"
        collections = ["hls"]
        query = {"hls:product": {"eq": hls_product}, "eo:cloud_cover": {"lt": float(max_cloud)}}
    else:
        collections = [coll_id]
        query = {"eo:cloud_cover": {"lt": float(max_cloud)}}

    search = catalog.search(
        collections=collections,
        intersects=aoi_geometry.__geo_interface__,
        datetime=f"{date_start}/{date_end}",
        query=query,
        max_items=500,
    )
    try:
        items = list(search.items())
    except Exception:
        items = list(search.get_items())
    if not items:
        return []

    def _sort_key(it):
        d = dt.datetime.fromisoformat(it.properties["datetime"].replace("Z", ""))
        month_bias = 0 if (preferred_months and d.month in preferred_months) else 1
        return (month_bias, it.properties.get("eo:cloud_cover", 100), d)

    items.sort(key=_sort_key)

    selected = []
    used_days: set[Tuple[int, int]] = set()
    for it in items:
        d = dt.datetime.fromisoformat(it.properties["datetime"].replace("Z", ""))
        yearday = (d.year, d.timetuple().tm_yday)
        if any(abs(d.timetuple().tm_yday - ud) <= min_day_gap and d.year == uy for (uy, ud) in used_days):
            continue
        selected.append(it)
        used_days.add(yearday)
        if len(selected) >= n_samples:
            break
    signed_items = [pc.sign(it).to_dict() for it in selected]
    return signed_items


def download_hls_samples(
    out_dir: Path,
    aoi_geometry,
    date_start: str,
    date_end: str,
    collection: str = "hls2-s30",
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    n_samples: int = 4,
    max_cloud: int = 20,
    preferred_months: Optional[Sequence[int]] = tuple(range(4, 11)),
    bands: Sequence[str] = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"),
    resolution: int = 30,
    name_prefix: str = "HLS",
    verbose: bool = False,
    min_day_gap: int = 10,
    dry_run: bool = False,
    items_json_out: Optional[Path] = None,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coll_id = _resolve_collection_id(collection)
    if verbose:
        print("[hls] collection:", coll_id)
    signed_items = _search_and_select_items(
        stac_url=stac_url,
        collection=collection,
        aoi_geometry=aoi_geometry,
        date_start=date_start,
        date_end=date_end,
        max_cloud=max_cloud,
        preferred_months=preferred_months,
        n_samples=n_samples,
        min_day_gap=min_day_gap,
    )
    if verbose:
        print("[hls] selected:", len(signed_items))
    if not signed_items:
        return []
    if items_json_out is not None:
        try:
            meta = [
                {
                    "id": it.get("id") or it.get("properties", {}).get("id"),
                    "datetime": it["properties"].get("datetime"),
                    "hls:tile_id": it["properties"].get("hls:tile_id"),
                    "mgrs:tile": it["properties"].get("mgrs:tile"),
                }
                for it in signed_items
            ]
            items_json_out = Path(items_json_out)
            items_json_out.parent.mkdir(parents=True, exist_ok=True)
            items_json_out.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass
    suffix = "S30" if coll_id.endswith("s30") else "L30"

    written: List[Path] = []
    for it_signed in signed_items:
        props = it_signed["properties"]
        tile_id = props.get("hls:tile_id") or props.get("mgrs:tile")
        if not tile_id:
            item_id = it_signed.get("id") or props.get("id")
            if isinstance(item_id, str):
                parts = item_id.split(".")
                if len(parts) >= 3 and parts[2].startswith("T"):
                    tile_id = parts[2]
        if not tile_id:
            tile_id = "TXXXXXX"
        dt_iso = props["datetime"]
        dt_obj = dt.datetime.fromisoformat(dt_iso.replace("Z", ""))
        yyyyddd = f"{dt_obj.year}{dt_obj.timetuple().tm_yday:03d}"
        hhmmss = dt_obj.strftime("%H%M%S")
        out_name = f"{name_prefix}_HLS.{suffix}.{tile_id}.{yyyyddd}T{hhmmss}.v2.0_cropped.tif"
        out_path = out_dir / out_name

        try:
            if dry_run:
                if verbose:
                    print("[hls] plan: would write", out_path)
                written.append(out_path)
                continue
            if verbose:
                print("[hls] writing:", out_path)
            single = {"type": "FeatureCollection", "features": [it_signed]}
            da = stack(
                single,
                assets=list(bands),
                chunksize=1024,
                resolution=resolution,
                bounds=aoi_geometry.bounds,
                dtype="uint16",
                rescale=False,
                fill_value=0,
            )
            da = da.compute().transpose("band", "y", "x")
            da = da.assign_coords(band=("band", list(bands)))
            da = da.rio.write_crs("EPSG:4326", inplace=True).rio.reproject_match(da)
            da_clipped = da.rio.clip([aoi_geometry.__geo_interface__], crs="EPSG:4326", drop=False)
            da_clipped.rio.to_raster(
                out_path,
                dtype="uint16",
                compress="deflate",
                predictor=2,
                tiled=True,
                BIGTIFF="IF_SAFER",
            )
            written.append(out_path)
        except Exception as e:
            print("[hls] ERROR writing", out_path, "->", repr(e))

    return written


def _prompt(prompt_text: str, default: Optional[str] = None) -> str:
    txt = input(f"{prompt_text} [{default if default is not None else ''}]: ").strip()
    return txt or (default or "")

def _examples_text() -> str:
    return (
        "Examples:\n"
        "  - Search and download 4 HLSS30 samples for Santa Barbara (7 km buffer):\n"
        "    python -m geogfm.utils.hls_downloads \\\n+\n      --start 2018-01-01 --end 2020-12-31 \\\n+\n      --collection hls2-s30 --center-lat 34.4138 --center-lon -119.8489 --buffer-km 7\n\n"
        "  - Dry-run with config preview and item list (no downloads):\n"
        "    python -m geogfm.utils.hls_downloads --dry-run --print-config --print-items \\\n+\n      --start 2018-01-01 --end 2018-12-31 --center-lat 34.4138 --center-lon -119.8489 --buffer-km 5\n\n"
        "  - Use an AOI GeoJSON instead of center/buffer:\n"
        "    python -m geogfm.utils.hls_downloads --aoi-geojson path/to/aoi.geojson --start 2019-01-01 --end 2019-12-31\n"
    )

def _console() -> Optional[Console]:
    try:
        return Console() if Console is not None else None
    except Exception:
        return None


def _print_banner(cons: Optional[Console]) -> None:
    if cons is None:
        return
    title = Text("HLS Samples Wizard", style="bold magenta")
    subtitle = Text("Microsoft Planetary Computer · Cropped GeoTIFF outputs", style="cyan")
    cons.print(Panel.fit(Text.assemble(title, "\n", subtitle), border_style="magenta", title="GeoAI", subtitle="v2"))


def _prompt_rich(prompt_text: str, default: Optional[str] = None) -> str:
    cons = _console()
    if cons is not None and Prompt is not None:
        return Prompt.ask(f"[bold cyan]{prompt_text}[/]", default=default or "")
    return _prompt(prompt_text, default)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Download cropped HLS v2 scenes from Microsoft Planetary Computer",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=_examples_text(),
    )
    p.add_argument("--stac-url", default="https://planetarycomputer.microsoft.com/api/stac/v1")
    p.add_argument("--collection", default="hls2-s30", help="hls2-s30 | hls2-l30 | HLSS30 | HLSL30")
    p.add_argument("--start", required=False, help="YYYY-MM-DD")
    p.add_argument("--end", required=False, help="YYYY-MM-DD")
    p.add_argument("--out-dir", default="./data/hls_downloads")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-cloud", type=int, default=20)
    p.add_argument("--bands", default="B02,B03,B04,B05,B06,B07,B08,B11,B12")
    p.add_argument("--resolution", type=int, default=30)
    p.add_argument("--name-prefix", default="HLS")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Plan and print without downloading")
    p.add_argument("--print-config", action="store_true", help="Print resolved configuration before running")
    p.add_argument("--print-items", action="store_true", help="List selected items before writing")
    p.add_argument("--min-day-gap", type=int, default=10, help="Minimum separation in days between selected scenes")
    p.add_argument(
        "--preferred-months",
        type=str,
        default="4-10",
        help="Comma list (e.g., 4,5,6,7,8,9,10) or a range '4-10'; use 'none' to disable",
    )
    p.add_argument("--items-json-out", type=str, default=None, help="Path to write selected item metadata (JSON)")

    aoi = p.add_mutually_exclusive_group(required=False)
    aoi.add_argument("--aoi-geojson", type=str, default=None, help="Path to GeoJSON polygon/feature/featurecollection")
    aoi.add_argument("--center-lat", type=float, default=None)
    p.add_argument("--center-lon", type=float, default=None)
    p.add_argument("--buffer-km", type=float, default=None)

    args = p.parse_args(argv)

    if args.interactive:
        cons = _console()
        _print_banner(cons)
        if cons is not None:
            cons.print(Markdown("""
            Welcome! This guided wizard will help you fetch Harmonized Landsat & Sentinel-2 (HLS) scenes
            from the Microsoft Planetary Computer, clipped to your area of interest (AOI), and saved as
            compact GeoTIFFs suitable for teaching and experimentation.

            Press Enter to accept defaults. You can also exit at any time with Ctrl+C.
            """))

        # Dates and collection
        if not args.start:
            args.start = _prompt_rich("Start date (YYYY-MM-DD)", "2018-01-01")
        if not args.end:
            args.end = _prompt_rich("End date (YYYY-MM-DD)", "2020-12-31")
        coll_in = _prompt_rich("Collection (hls2-s30 | hls2-l30 | HLSS30 | HLSL30)", str(args.collection))
        args.collection = coll_in or args.collection

        # AOI choice
        if args.aoi_geojson is None and (args.center_lat is None or args.center_lon is None or args.buffer_km is None):
            mode = _prompt_rich("AOI mode (geojson | center)", "center")
            if mode.lower().startswith("geo"):
                args.aoi_geojson = _prompt_rich("Path to AOI GeoJSON", "")
            else:
                args.center_lat = float(_prompt_rich("Center latitude", str(34.4138)))
                args.center_lon = float(_prompt_rich("Center longitude", str(-119.8489)))
                args.buffer_km = float(_prompt_rich("Buffer radius (km)", str(10)))

        # Sampling / filters
        args.n_samples = int(_prompt_rich("Number of samples to save", str(args.n_samples)))
        args.max_cloud = int(_prompt_rich("Max cloud cover (%)", str(args.max_cloud)))
        args.min_day_gap = int(_prompt_rich("Min day gap between samples", str(args.min_day_gap or 10)))
        args.preferred_months = _prompt_rich("Preferred months (e.g., 4-10, 6,7,8 or 'none')", "4-10")

        # Output
        args.out_dir = _prompt_rich("Output directory", str(args.out_dir))
        args.name_prefix = _prompt_rich("Filename prefix", str(args.name_prefix))

        if cons is not None and Confirm is not None:
            proceed = Confirm.ask("Proceed with search and download?", default=True)
            if not proceed:
                cons.print("Aborted by user.", style="yellow")
                return 0

    # Validate AOI
    if args.aoi_geojson:
        aoi_geom = _load_aoi_from_geojson(Path(args.aoi_geojson))
    else:
        if args.center_lat is None or args.center_lon is None or args.buffer_km is None:
            p.error("Provide --aoi-geojson or all of --center-lat/--center-lon/--buffer-km")
        aoi_geom = _build_aoi_from_center(args.center_lat, args.center_lon, args.buffer_km)

    bands = tuple([b.strip() for b in args.bands.split(",") if b.strip()])

    # Parse preferred months
    preferred_months: Optional[Sequence[int]]
    pm_str = (args.preferred_months or "").strip().lower()
    if pm_str in {"none", "", "false", "0"}:
        preferred_months = None
    elif "-" in pm_str and "," not in pm_str:
        a, b = pm_str.split("-", 1)
        try:
            preferred_months = tuple(range(int(a), int(b) + 1))
        except Exception:
            preferred_months = tuple(range(4, 11))
    else:
        try:
            preferred_months = tuple(int(x) for x in pm_str.split(",") if x)
        except Exception:
            preferred_months = tuple(range(4, 11))

    cons = _console()
    if cons is not None and args.print_config:
        table = Table(title="HLS Download Configuration", box=box.SIMPLE_HEAVY if box else None)
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Value", style="white")
        table.add_row("Collection", str(args.collection))
        table.add_row("Date range", f"{args.start or '2018-01-01'} → {args.end or '2020-12-31'}")
        table.add_row("n_samples", str(args.n_samples))
        table.add_row("max_cloud", str(args.max_cloud))
        table.add_row("preferred_months", str(list(preferred_months) if preferred_months else None))
        table.add_row("min_day_gap", str(args.min_day_gap))
        table.add_row("bands", ",".join(bands))
        table.add_row("resolution", str(args.resolution))
        table.add_row("out_dir", str(Path(args.out_dir).resolve()))
        table.add_row("dry_run", str(bool(args.dry_run)))
        cons.print(Panel.fit(table, title="Plan", border_style="green"))

    # Optionally print selected items prior to downloads
    if args.print_items or args.dry_run:
        sel = _search_and_select_items(
            stac_url=args.stac_url,
            collection=args.collection,
            aoi_geometry=aoi_geom,
            date_start=args.start or "2018-01-01",
            date_end=args.end or "2020-12-31",
            max_cloud=args.max_cloud,
            preferred_months=preferred_months,
            n_samples=args.n_samples,
            min_day_gap=int(args.min_day_gap),
        )
        cons = _console()
        if cons is not None and sel:
            tbl = Table(title="Selected Items", box=box.MINIMAL_HEAVY_HEAD if box else None)
            tbl.add_column("#", justify="right", style="bold")
            tbl.add_column("ID")
            tbl.add_column("Datetime")
            tbl.add_column("Tile")
            for idx, it in enumerate(sel, 1):
                props = it.get("properties", {})
                tile = props.get("hls:tile_id") or props.get("mgrs:tile") or "?"
                tbl.add_row(str(idx), str(it.get("id")), str(props.get("datetime")), str(tile))
            cons.print(tbl)
        elif sel:
            print("Selected items:")
            for it in sel:
                props = it.get("properties", {})
                tile = props.get("hls:tile_id") or props.get("mgrs:tile") or "?"
                print(" -", it.get("id"), props.get("datetime"), tile)

    # Run with spinner if rich is available
    cons = _console()
    if cons is not None and Progress is not None and not args.dry_run:
        with Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=cons,
        ) as progress:
            task = progress.add_task("Searching and downloading HLS scenes...", total=None)
            saved = download_hls_samples(
                out_dir=Path(args.out_dir),
                aoi_geometry=aoi_geom,
                date_start=args.start or "2018-01-01",
                date_end=args.end or "2020-12-31",
                collection=args.collection,
                stac_url=args.stac_url,
                n_samples=args.n_samples,
                max_cloud=args.max_cloud,
                bands=bands,
                resolution=args.resolution,
                name_prefix=args.name_prefix,
                verbose=True,
                min_day_gap=int(args.min_day_gap),
                dry_run=bool(args.dry_run),
                items_json_out=Path(args.items_json_out) if args.items_json_out else None,
            )
            progress.update(task, completed=1)
    else:
        saved = download_hls_samples(
            out_dir=Path(args.out_dir),
            aoi_geometry=aoi_geom,
            date_start=args.start or "2018-01-01",
            date_end=args.end or "2020-12-31",
            collection=args.collection,
            stac_url=args.stac_url,
            n_samples=args.n_samples,
            max_cloud=args.max_cloud,
            bands=bands,
            resolution=args.resolution,
            name_prefix=args.name_prefix,
            verbose=True,
            min_day_gap=int(args.min_day_gap),
            dry_run=bool(args.dry_run),
            items_json_out=Path(args.items_json_out) if args.items_json_out else None,
        )
    if cons is not None:
        if not saved:
            cons.print(Panel.fit(Text("No items matched your query. Try loosening dates, clouds, or AOI size.", style="yellow"), title="No Results", border_style="red"))
        else:
            tbl = Table(title="Outputs", box=box.MINIMAL_HEAVY_HEAD if box else None)
            tbl.add_column("#", justify="right", style="bold")
            tbl.add_column("File", overflow="fold")
            for idx, pth in enumerate(saved, 1):
                tbl.add_row(str(idx), str(pth))
            cons.print(tbl)
            action_word = "Planned" if args.dry_run else "Saved"
            cons.print(Panel.fit(Text(f"{action_word} {len(saved)} files → {Path(args.out_dir).resolve()}", style="bold green"), border_style="green"))
    else:
        print(f"Saved {len(saved)} files to {Path(args.out_dir).resolve()}")
        for pth in saved:
            print(" -", pth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

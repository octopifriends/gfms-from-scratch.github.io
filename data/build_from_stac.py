#!/usr/bin/env python3
"""
build_from_stac.py

Query a STAC API (default: Microsoft Planetary Computer), build a reproducible
scene manifest + scene-level splits for a GFM course/book.

Outputs (under --out-dir, default data/out):
  meta/scenes.parquet
  meta/splits/{train,val,test}_scenes.txt
  CHECKSUMS.md

Use --dryrun to preview what would be written without creating files.

Examples
--------
BBox over inland Central Coast (Sentinel-2 L2A), June 2020:
  python data/build_from_stac.py \
    --collections sentinel-2-l2a \
    --bbox -120.30 34.55 -119.95 34.80 \
    --start 2020-06-01 --end 2020-06-30 \
    --cloud-cover-max 20 --max-scenes-per-aoi 12 --dryrun

Multiple AOIs from GeoJSON, stratify by month and AOI:
  python data/build_from_stac.py \
    --collections sentinel-2-l2a \
    --aoi-file data/aois/central_coast_inland.geojson \
    --start 2020-06-01 --end 2020-09-30 \
    --cloud-cover-max 25 --max-scenes-per-aoi 30 \
    --stratify month aoi
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd
from pystac_client import Client
from shapely.geometry import mapping, box

# Optional MPC signing
try:
    import planetary_computer as pc
    HAVE_PC = True
except ImportError:
    HAVE_PC = False


# ---------- utils ----------

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path):
    out = root / "CHECKSUMS.md"
    with open(out, "w") as f:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "CHECKSUMS.md":
                f.write(f"{sha256(p)}  {p.relative_to(root)}\n")


def month_str(dt_iso: str) -> str:
    try:
        dt = datetime.fromisoformat((dt_iso or "").replace("Z", "+00:00"))
        return f"{dt.year}-{dt.month:02d}"
    except Exception:
        return "unknown"


def load_aoi_geometries(aoi_file: Optional[str], bbox: Optional[Iterable[float]]):
    """Return list[(aoi_id, geom_geojson_dict)]."""
    geoms = []
    if aoi_file:
        gj = json.loads(Path(aoi_file).read_text())
        feats = gj["features"] if "features" in gj else [gj]
        for i, feat in enumerate(feats):
            aoi_id = feat.get("properties", {}).get("aoi_id", f"aoi{i:03d}")
            geoms.append((aoi_id, feat["geometry"]))
    elif bbox:
        minx, miny, maxx, maxy = map(float, bbox)
        geoms.append(("aoi000", mapping(box(minx, miny, maxx, maxy))))
    else:
        raise ValueError("Provide either --aoi-file or --bbox.")
    return geoms


# ---------- core ----------

def stac_search(stac_url: str, collection: str, geom: dict, start: str, end: str,
                cloud_cover_max: Optional[float], limit: int):
    client = Client.open(stac_url)
    query = {}
    if cloud_cover_max is not None:
        query["eo:cloud_cover"] = {"lt": float(cloud_cover_max)}
    search = client.search(
        collections=[collection],
        intersects=geom,
        datetime=f"{start}/{end}",
        query=query,
        max_items=limit,
    )
    return list(search.items())


def build_manifest_from_items(items, aoi_id: str, sign_assets: bool) -> pd.DataFrame:
    rows: List[dict] = []
    for it in items:
        props = it.properties or {}
        scene_id = it.id
        dt = props.get("datetime") or props.get("start_datetime") or ""
        epsg = props.get("proj:epsg")
        gsd = props.get("gsd")
        platform = props.get("platform")
        instruments = ";".join(props.get("instruments", []) or [])
        sun_elev = props.get("view:sun_elevation")
        sun_azi = props.get("view:sun_azimuth")
        cloud = props.get("eo:cloud_cover")

        # Default (unsigned) asset hrefs
        assets = {k: v.href for k, v in it.assets.items()}

        if sign_assets:
            if not HAVE_PC:
                print(
                    "[WARN] --sign-assets requested but planetary-computer is not installed. Skipping signing.")
            else:
                try:
                    signed = pc.sign(it)
                    assets = {k: v.href for k, v in signed.assets.items()}
                except Exception as e:
                    print(f"[WARN] Failed to sign assets for {scene_id}: {e}")

        rows.append(dict(
            aoi_id=aoi_id,
            scene_id=scene_id,
            collection=it.collection_id,
            datetime=dt,
            month=month_str(dt),
            epsg=epsg,
            gsd=gsd,
            platform=platform,
            instruments=instruments,
            cloud_cover=cloud,
            sun_elevation=sun_elev,
            sun_azimuth=sun_azi,
            bbox=json.dumps(it.bbox),
            geometry=json.dumps(it.geometry),
            assets=json.dumps(assets),
        ))
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, seed=1337, frac_train=0.70, frac_val=0.15,
                     strata_cols: Optional[List[str]] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = None

    if not strata_cols:
        ids = df["scene_id"].unique().tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(n * frac_train)
        n_val = int(n * frac_val)
        split = {sid: "train" for sid in ids[:n_tr]}
        split.update({sid: "val" for sid in ids[n_tr:n_tr + n_val]})
        split.update({sid: "test" for sid in ids[n_tr + n_val:]})
        df["split"] = df["scene_id"].map(split)
        return df

    parts = []
    for _, grp in df.groupby(strata_cols, dropna=False):
        ids = grp["scene_id"].unique().tolist()
        rng.shuffle(ids)
        n = len(ids)
        
        # More robust allocation that ensures val split gets scenes
        if n == 1:
            # Single scene goes to train
            n_tr, n_val = 1, 0
        elif n == 2:
            # Two scenes: one train, one test
            n_tr, n_val = 1, 0
        elif n == 3:
            # Three scenes: two train, one val
            n_tr, n_val = 2, 1
        else:
            # Four or more scenes: use proportional allocation with minimum val=1
            n_tr = max(1, int(n * frac_train))
            n_val = max(1, int(n * frac_val))
            # Ensure we don't exceed total scenes
            if n_tr + n_val >= n:
                n_tr = n - 2
                n_val = 1
        
        split = {sid: "train" for sid in ids[:n_tr]}
        split.update({sid: "val" for sid in ids[n_tr:n_tr + n_val]})
        split.update({sid: "test" for sid in ids[n_tr + n_val:]})
        g = grp.copy()
        g["split"] = g["scene_id"].map(split)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def stratified_downsample(df: pd.DataFrame, target: int, seed: int, strata_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Downsample df to ~target rows, proportional per stratum, deterministic."""
    if target <= 0 or len(df) <= target:
        return df.copy()

    rng = np.random.default_rng(seed)
    strata = strata_cols if strata_cols else ["month", "aoi_id"]

    groups = []
    total = len(df)
    for key, g in df.groupby(strata, dropna=False):
        k = max(1, int(round(len(g) * target / total)))
        idx = np.arange(len(g))
        rng.shuffle(idx)
        groups.append(g.iloc[idx[:min(k, len(g))]])

    out = pd.concat(groups, ignore_index=True).drop_duplicates("scene_id")

    if len(out) > target:
        idx = np.arange(len(out))
        rng.shuffle(idx)
        out = out.iloc[idx[:target]].copy()

    return out


def write_scene_splits(df: pd.DataFrame, out_meta: Path):
    ensure_dir(out_meta / "splits")
    for sp in ("train", "val", "test"):
        ids = df.loc[df["split"] == sp, "scene_id"].drop_duplicates().tolist()
        with open(out_meta / "splits" / f"{sp}_scenes.txt", "w") as f:
            for sid in ids:
                f.write(f"{sid}\n")


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description="Build STAC scene manifest + splits.")
    p.add_argument(
        "--stac-url", default="https://planetarycomputer.microsoft.com/api/stac/v1")
    p.add_argument("--collections", nargs="+", default=["sentinel-2-l2a"])
    # AOI
    p.add_argument("--aoi-file", type=str, default=None,
                   help="GeoJSON with Polygon/MultiPolygon feature(s)")
    p.add_argument("--bbox", nargs=4, type=float, default=None,
                   help="minx miny maxx maxy (lon/lat)")
    # Time & filters
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--cloud-cover-max", type=float,
                   default=None, help="e.g., 20 for <=20%% clouds")
    p.add_argument("--max-scenes-per-aoi", type=int, default=50)
    p.add_argument(
        "--target-total-scenes", type=int, default=0,
        help="Optional global cap on total scenes (downsampled with stratification before splitting). 0 = disabled"
    )
    # Splits
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--frac-train", type=float, default=0.70)
    p.add_argument("--frac-val", type=float, default=0.15)
    p.add_argument("--stratify", nargs="*", default=[],
                   help='choose among "month" and "aoi"')
    # Assets/signing
    p.add_argument("--sign-assets", action="store_true",
                   help="Sign MPC asset URLs (requires planetary-computer)")
    # I/O
    p.add_argument("--out-dir", default="data/out")
    p.add_argument("--dryrun", action="store_true",
                   help="Preview: search/split but do not write files")
    args = p.parse_args()

    # Load AOIs
    aois = load_aoi_geometries(args.aoi_file, args.bbox)

    # Query per AOI & collection
    all_df = []
    for aoi_id, geom in aois:
        for coll in args.collections:
            items = stac_search(
                stac_url=args.stac_url,
                collection=coll,
                geom=geom,
                start=args.start,
                end=args.end,
                cloud_cover_max=args.cloud_cover_max,
                limit=args.max_scenes_per_aoi,
            )
            if not items:
                continue
            df = build_manifest_from_items(
                items, aoi_id=aoi_id, sign_assets=args.sign_assets)
            all_df.append(df)

    if not all_df:
        raise SystemExit("No scenes found. Adjust AOI/date/cloud filters.")

    scenes = pd.concat(all_df, ignore_index=True)

    # Optional stratified downsampling
    if args.target_total_scenes and args.target_total_scenes > 0:
        strata_cols = []
        if "month" in args.stratify: strata_cols.append("month")
        if "aoi" in args.stratify: strata_cols.append("aoi_id")
        before = len(scenes)
        scenes = stratified_downsample(
            scenes,
            target=args.target_total_scenes,
            seed=args.seed,
            strata_cols=strata_cols if strata_cols else ["month", "aoi_id"],
        )
        print(f"[INFO] Downsampled scenes: {before} → {len(scenes)} using target_total_scenes={args.target_total_scenes}")

    # Stratified splits
    strata_cols = []
    if "month" in args.stratify:
        strata_cols.append("month")
    if "aoi" in args.stratify:
        strata_cols.append("aoi_id")
    scenes = stratified_split(
        scenes,
        seed=args.seed,
        frac_train=args.frac_train,
        frac_val=args.frac_val,
        strata_cols=strata_cols if strata_cols else None,
    )

    # DRYRUN preview
    if args.dryrun:
        print("[DRYRUN] STAC:", args.stac_url)
        print("AOIs:", len(aois), "| Collections:", ", ".join(args.collections))
        print(
            f"Date range: {args.start} → {args.end} | Cloud ≤ {args.cloud_cover_max}% | Max/ AOI: {args.max_scenes_per_aoi}")
        tot = scenes["scene_id"].nunique()
        split_counts = scenes.groupby("split")["scene_id"].nunique().to_dict()
        print(f"Found total {tot} unique scenes | Split:", ", ".join(
            f"{k}={v}" for k, v in split_counts.items()))
        
        # Per-AOI counts
        aoi_counts = scenes.groupby("aoi_id")["scene_id"].nunique().sort_values(ascending=False)
        print(f"\nScenes per AOI:")
        for aoi, count in aoi_counts.items():
            print(f"  - {aoi}: {count} scenes")
        
        # Per-month counts
        month_counts = scenes.groupby("month")["scene_id"].nunique().sort_index()
        print(f"\nScenes per month:")
        for month, count in month_counts.items():
            print(f"  - {month}: {count} scenes")
        
        cols = ["aoi_id", "collection", "scene_id", "datetime", "cloud_cover"]
        print(f"\n--- PREVIEW: First 8 rows (of {len(scenes)} total rows) ---")
        print(scenes[cols].head(8).to_string(index=False))
        print("--- END PREVIEW ---")
        print("\n[DRYRUN] No files written. Run without --dryrun to save outputs.")
        return

    # Write artifacts
    out_dir = Path(args.out_dir)
    out_meta = out_dir / "meta"
    ensure_dir(out_meta)

    scenes.to_parquet(out_meta / "scenes.parquet", index=False)
    write_scene_splits(scenes, out_meta)
    write_checksums(out_dir)

    print(f"[OK] Wrote {len(scenes)} rows to {out_meta/'scenes.parquet'}")
    for sp in ("train", "val", "test"):
        nsp = scenes.query("split == @sp")["scene_id"].nunique()
        print(f"  - {sp}: {nsp} scenes")
    print(f"[OK] Checksums at {out_dir/'CHECKSUMS.md'}")


if __name__ == "__main__":
    main()

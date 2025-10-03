import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio

class GeoGFMDataset(Dataset):
    def __init__(self, root, split, meta_fields=("lat","lon","dt","cloud_frac")):
        self.root = Path(root)
        self.split = split
        self.meta = pd.read_parquet(self.root / "meta" / "patches.parquet")
        with open(self.root / "meta" / "splits" / f"{split}.txt") as f:
            ids = set([ln.strip() for ln in f if ln.strip()])
        self.meta = self.meta[self.meta["patch_id"].isin(ids)].reset_index(drop=True)
        self.meta_fields = meta_fields
        # simple encoders
        self._time_enc = lambda s: self._encode_time(s)  # iso8601 to [sin(day), cos(day)]
        self._stdz = lambda x, m, s: (x - m) / (s + 1e-6)

    def __len__(self):
        return len(self.meta)

    def _encode_time(self, iso):
        # Expect "YYYY-MM-DDTHH:MM:SSZ"
        if isinstance(iso, float) or iso is None or iso == "YYYY-MM-DDTHH:MM:SSZ":
            return np.array([0., 1.], dtype=np.float32)
        from datetime import datetime
        dt = datetime.fromisoformat(iso.replace("Z","+00:00"))
        doy = dt.timetuple().tm_yday
        ang = 2*np.pi*(doy/365.0)
        return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)

    def _read_tif_chw(self, path):
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # [C,H,W]
        return arr

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid, scene = row["patch_id"], row["scene_id"]
        s2_path = self.root / "images" / self.split / scene / f"{pid}_S2.tif"
        s1_path = self.root / "images" / self.split / scene / f"{pid}_S1.tif"

        s2 = self._read_tif_chw(s2_path)  # [10,H,W]
        s1 = self._read_tif_chw(s1_path)  # [2,H,W]
        x = np.concatenate([s2, s1], axis=0)  # [12,H,W]

        # Metadata tensor
        meta_vals = []
        for k in self.meta_fields:
            v = row.get(k, np.nan)
            if k == "dt":
                meta_vals.append(self._time_enc(v))
            else:
                meta_vals.append(np.array([np.float32(v if np.isfinite(v) else 0.0)]))
        m = np.concatenate(meta_vals).astype(np.float32)  # shape M

        # Label
        y_path = self.root.parent / "core" / "labels" / self.split / f"{pid}.npy"
        if not y_path.exists():
            # fallback to dummy label
            y = np.array([0], dtype=np.int64)
        else:
            y = np.load(y_path).astype(np.int64)

        return (
            torch.from_numpy(x),            # [12,H,W]
            torch.from_numpy(m),            # [M]
            torch.from_numpy(y).squeeze(0)  # scalar class id
        )
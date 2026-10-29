# data_reader.py
from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from smart_open import open as sopen

# NEW: for listing files
import boto3
from botocore.config import Config as BotoConfig
from urllib.parse import urlparse


# ----------------------------- Utilities ---------------------------------- #

def _parse_channel_number(ch_key: str) -> int:
    return int("".join([c for c in ch_key if c.isdigit()]))

def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """
    's3://bucket/prefix/sub/' -> ('bucket', 'prefix/sub/')
    Accepts with/without trailing slash. Returns prefix possibly empty ''.
    """
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Not an S3 URL: {s3_url}")
    p = urlparse(s3_url)
    bucket = p.netloc
    # urlparse gives path starting with '/', strip it
    prefix = p.path.lstrip("/")
    return bucket, prefix

def _iter_s3_keys(bucket: str, prefix: str, suffix: Optional[str] = None) -> Iterable[str]:
    """
    Generator over object keys under bucket/prefix (recursive).
    Uses pagination; yields keys (not URLs).
    """
    client = boto3.client("s3", config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}))
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if suffix and not key.endswith(suffix):
                continue
            yield key
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break


# ----------------------------- Base Interface ----------------------------- #

class BaseDataReader(ABC):
    """Common interface every turbine data reader must implement."""

    @abstractmethod
    def read(self, file_name: str) -> pd.DataFrame:
        """Read a single remote file into a tidy DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def list_files(
        self,
        *,
        recursive: bool = True,
        startswith: Optional[str] = None,
        endswith: Optional[str] = None,
        contains: Optional[str] = None,
        limit: Optional[int] = None,
        return_full_urls: bool = True,
    ) -> List[str]:
        """
        List files under this reader's s3_prefix.

        Filters (all optional) are applied to the KEY (not the URL):
          - startswith, endswith, contains
        If limit is provided, stops early.
        """


# ----------------------------- Structure-1 Reader --------------------------- #

@dataclass
class Structure1Reader(BaseDataReader):
    s3_prefix: str
    sensor_config: Dict[str, Sequence[str]]

    def read(self, file_name: str) -> pd.DataFrame:
        url = f"{self.s3_prefix}{file_name}"
        with sopen(url, "r") as fin:
            data = json.loads(fin.read())

        base = pd.DataFrame(
            {"time": data["time"], "channel": data["channel"], "signals": data["signals"]}
        )
        sig = pd.DataFrame(base["signals"].tolist(), index=base.index)
        sig.columns = [f"s{i+1}" for i in range(sig.shape[1])]
        df = pd.concat([base[["time", "channel"]], sig], axis=1)

        final = pd.DataFrame()
        first_ch = list(self.sensor_config.keys())[0]
        first_num = _parse_channel_number(first_ch)
        tdf = df[df["channel"] == first_num]
        if tdf.empty:
            warnings.warn(f"No rows for first channel {first_num} in {url}")
            return pd.DataFrame()

        final["time"] = tdf["time"].values

        for ch_key, sensors in self.sensor_config.items():
            ch_num = _parse_channel_number(ch_key)
            cdf = df[df["channel"] == ch_num]
            if cdf.empty:
                for s in sensors:
                    final[f"{ch_key}{s}"] = pd.Series([pd.NA] * len(final), dtype="float64")
                continue
            for s in sensors:
                final[f"{ch_key}{s}"] = pd.to_numeric(cdf.get(s, pd.NA), errors="coerce").values

        return final

    def list_files(
        self,
        *,
        recursive: bool = True,
        startswith: Optional[str] = None,
        endswith: Optional[str] = None,
        contains: Optional[str] = None,
        limit: Optional[int] = None,
        ) -> List[str]:
        """
        List all files (basenames only) under this reader's s3_prefix.
        Example output: ['Structure1_data_20230714140041.gz', 'Structure1_data_20230714142041.gz']
        """
        bucket, prefix = _parse_s3_url(self.s3_prefix)
        out: List[str] = []

        for key in _iter_s3_keys(bucket, prefix):
            name = key[len(prefix):] if key.startswith(prefix) else key

            # Skip nested folders if not recursive
            if not recursive and "/" in name.strip("/"):
                continue

            # Filters
            if startswith and not name.startswith(startswith):
                continue
            if endswith and not name.endswith(endswith):
                continue
            if contains and (contains not in name):
                continue

            # Use only the basename (after last '/')
            basename = name.split("/")[-1]
            out.append(basename)

            if limit and len(out) >= limit:
                break

        return out



# ----------------------------- Structure-3/7 Reader ------------------------- #

@dataclass
class PeakStructureReader(BaseDataReader):
    s3_prefix: str
    robust_reading: bool = True
    pair_tolerance: str = "1ms"
    deduplicate_index: Optional[str] = None

    rename_map_c1: Dict[str, str] = field(default_factory=lambda: {
        "peak1": "B_flap_nearsided",
        "peak2": "B_edge_nearsided",
        "peak3": "B_flap_farsided",
        "peak4": "B_edge_farsided",
        "peak5": "A_flap_nearsided",
        "peak6": "A_edge_nearsided",
    })
    rename_map_c2: Dict[str, str] = field(default_factory=lambda: {
        "peak1": "A_flap_farsided",
        "peak2": "A_edge_farsided",
        "peak3": "C_flap_nearsided",
        "peak4": "C_edge_farsided",
        "peak5": "C_flap_farsided",
        "peak6": "C_edge_farsided",
    })
    desired_order: List[str] = field(default_factory=lambda: [
        "B_flap_nearsided", "B_edge_nearsided", "B_flap_farsided", "B_edge_farsided",
        "A_flap_nearsided", "A_edge_nearsided", "A_flap_farsided", "A_edge_farsided",
        "C_flap_nearsided", "C_edge_farsided", "C_flap_farsided", "C_edge_farsided",
    ])

    def read(self, file_name: str) -> pd.DataFrame:
        url = f"{self.s3_prefix}{file_name}"

        # We'll try two passes: (A) assume the file HAS a header, (B) assume NO header.
        def _read_csv(pass_has_header: bool) -> pd.DataFrame:
            # Use python engine + sep=None to sniff delimiters; tolerate bad lines & comments
            if pass_has_header:
                return pd.read_csv(
                    sopen(url, "r"),
                    engine="python",
                    sep=None,            # infer delimiter
                    header=0,            # use first row as header
                    comment="#",
                    on_bad_lines="skip",
                )
            else:
                headers = ["timestamp", "fiber_id"] + [f"peak{i}" for i in range(1, 16)]
                return pd.read_csv(
                    sopen(url, "r"),
                    engine="python",
                    sep=None,
                    header=None,
                    names=headers,       # force names if no header
                    comment="#",
                    on_bad_lines="skip",
                )

        def _parse_timestamp(ts: pd.Series) -> pd.Series:
            # If numeric -> determine seconds vs milliseconds
            # If string -> let pandas parse
            # Return pd.DatetimeIndex-like Series
            s = ts.copy()
            # try numeric first
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().mean() > 0.9:  # mostly numeric
                mx = numeric.max()
                # Heuristic: >1e12 implies ms; >1e9 implies sec
                if mx > 1e12:
                    return pd.to_datetime(numeric, unit="ms", errors="coerce")
                elif mx > 1e9:
                    return pd.to_datetime(numeric, unit="s", errors="coerce")
                else:
                    # maybe already in ns or just small ints -> fall back to generic
                    return pd.to_datetime(numeric, errors="coerce", utc=False)
            # else, treat as string datetime
            return pd.to_datetime(s, errors="coerce", utc=False)

        # ---- Try reading with header, then without header ----
        df = pd.DataFrame()
        for has_header in (True, False):
            try:
                df = _read_csv(has_header)
                if not df.empty:
                    break
            except Exception:
                continue

        if df.empty:
            warnings.warn(f"Could not read or parse {url}")
            return pd.DataFrame()

        # Normalize expected columns (we tolerate different header names)
        # Find a plausible timestamp column
        ts_candidates = [c for c in df.columns if str(c).lower() in ("timestamp", "time", "datetime", "date")]
        if not ts_candidates:
            # assume first column is timestamp if nothing obvious
            ts_col = df.columns[0]
        else:
            ts_col = ts_candidates[0]

        # Fiber ID column
        fid_candidates = [c for c in df.columns if str(c).lower() in ("fiber_id", "fiber", "fiberid", "fid")]
        if not fid_candidates:
            # if missing fiber_id, create a default (all 1)
            df["fiber_id"] = 1
            fid_col = "fiber_id"
        else:
            fid_col = fid_candidates[0]

        # Parse timestamp robustly (no pre-filtering by regex)
        df[ts_col] = _parse_timestamp(df[ts_col])
        df = df.dropna(subset=[ts_col])

        if df.empty:
            warnings.warn(f"No valid timestamps found in {url}")
            return pd.DataFrame()

        # Ensure numeric peaks & fiber_id
        for i in range(1, 16):
            col = f"peak{i}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df[fid_col] = pd.to_numeric(df[fid_col], errors="coerce")
        df = df.dropna(subset=[fid_col])

        # Split c1/c2 as before
        df = df.rename(columns={ts_col: "timestamp", fid_col: "fiber_id"})
        df = df.set_index("timestamp").sort_index()

        df_c1 = df[df["fiber_id"] == 1].copy()
        df_c2 = df[df["fiber_id"] == 2].copy()

        # Apply your rename maps
        df_c1.rename(columns={k: v for k, v in self.rename_map_c1.items() if k in df_c1.columns}, inplace=True)
        df_c2.rename(columns={k: v for k, v in self.rename_map_c2.items() if k in df_c2.columns}, inplace=True)

        cols_c1 = [v for v in self.rename_map_c1.values() if v in df_c1.columns]
        cols_c2 = [v for v in self.rename_map_c2.values() if v in df_c2.columns]
        df_c1 = df_c1[cols_c1] if cols_c1 else pd.DataFrame(index=df_c1.index)
        df_c2 = df_c2[cols_c2] if cols_c2 else pd.DataFrame(index=df_c2.index)

        # From here on: unchanged logic (robust pairing or simple outer-join) ----
        if not self.robust_reading:
            out = pd.merge(df_c1, df_c2, how="outer", left_index=True, right_index=True).sort_index()
            final_cols = [c for c in self.desired_order if c in out.columns] + \
                        [c for c in out.columns if c not in self.desired_order]
            out = out[final_cols]
            if self.deduplicate_index in ("keep_first", "keep_last"):
                keep = "first" if self.deduplicate_index == "keep_first" else "last"
                out = out[~out.index.duplicated(keep=keep)]
            return out

        # --- robust pairing (same as your current implementation) ---
        c1 = df_c1.sort_index().reset_index().rename(columns={"timestamp": "ts_c1"})
        c2 = df_c2.sort_index().reset_index().rename(columns={"timestamp": "ts_c2"})
        if c1.empty and c2.empty:
            return pd.DataFrame()
        if c1.empty:
            res = c2.set_index("ts_c2"); res.index.name = "timestamp"
            return res.sort_index()
        if c2.empty:
            res = c1.set_index("ts_c1"); res.index.name = "timestamp"
            return res.sort_index()

        tol = pd.Timedelta(self.pair_tolerance)
        paired = pd.merge_asof(
            c1, c2, left_on="ts_c1", right_on="ts_c2",
            direction="nearest", tolerance=tol, suffixes=("_c1", "_c2")
        )
        matched = paired[~paired["ts_c2"].isna()].copy()
        matched["timestamp"] = matched["ts_c1"] + (matched["ts_c2"] - matched["ts_c1"]) / 2

        wanted = [c for c in (cols_c1 + cols_c2) if c in matched.columns]
        matched_df = matched[["timestamp"] + wanted].copy().set_index("timestamp")
        matched_df.index.name = "timestamp"

        unmatched_c1 = paired[paired["ts_c2"].isna()].copy()
        u1 = pd.DataFrame()
        if not unmatched_c1.empty:
            keep1 = ["ts_c1"] + [c for c in cols_c1 if c in unmatched_c1.columns]
            u1 = unmatched_c1[keep1].copy().rename(columns={"ts_c1": "timestamp"}).set_index("timestamp")
            u1.index.name = "timestamp"

        matched_c2_times = set(pd.to_datetime(matched["ts_c2"]).dropna().tolist())
        c2_only = c2[~c2["ts_c2"].isin(matched_c2_times)].copy()
        u2 = pd.DataFrame()
        if not c2_only.empty:
            keep2 = ["ts_c2"] + [c for c in cols_c2 if c in c2_only.columns]
            u2 = c2_only[keep2].copy().rename(columns={"ts_c2": "timestamp"}).set_index("timestamp")
            u2.index.name = "timestamp"

        pieces = [x for x in (matched_df, u1, u2) if x is not None and not x.empty]

        def _sanitize_unique_columns(x: pd.DataFrame) -> pd.DataFrame:
            if x.columns.duplicated().any():
                x = x.T.groupby(level=0).first().T
            return x

        pieces = [_sanitize_unique_columns(p.copy()) for p in pieces]

        union_cols: List[str] = [c for c in self.desired_order if any(c in p.columns for p in pieces)]
        extras: List[str] = []
        for p in pieces:
            for c in p.columns:
                if c not in union_cols and c not in extras:
                    extras.append(c)
        union_cols.extend(extras)

        pieces = [p.reindex(columns=union_cols) for p in pieces]
        out = pd.concat(pieces, axis=0, sort=False).sort_index()

        if self.deduplicate_index in ("keep_first", "keep_last"):
            keep = "first" if self.deduplicate_index == "keep_first" else "last"
            out = out[~out.index.duplicated(keep=keep)]

        return out


    def list_files(
        self,
        *,
        recursive: bool = True,
        startswith: Optional[str] = None,
        endswith: Optional[str] = None,
        contains: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        List all files (basenames only) under this reader's s3_prefix.
        Example output: ['Structure1_data_20230714140041.gz', 'Structure1_data_20230714142041.gz']
        """
        bucket, prefix = _parse_s3_url(self.s3_prefix)
        out: List[str] = []

        for key in _iter_s3_keys(bucket, prefix):
            name = key[len(prefix):] if key.startswith(prefix) else key

            # Skip nested folders if not recursive
            if not recursive and "/" in name.strip("/"):
                continue

            # Filters
            if startswith and not name.startswith(startswith):
                continue
            if endswith and not name.endswith(endswith):
                continue
            if contains and (contains not in name):
                continue

            # Use only the basename (after last '/')
            basename = name.split("/")[-1]
            out.append(basename)

            if limit and len(out) >= limit:
                break

        return out



# ----------------------------- Factory / Facade --------------------------- #

class DataReader:
    @staticmethod
    def for_turbine(
        turbine: str,
        *,
        s3_prefix: str,
        sensor_config: Optional[Dict[str, Sequence[str]]] = None,
        robust_reading: bool = True,
        pair_tolerance: str = "1ms",
        deduplicate_index: Optional[str] = None,
        rename_map_c1: Optional[Dict[str, str]] = None,
        rename_map_c2: Optional[Dict[str, str]] = None,
        desired_order: Optional[List[str]] = None,
    ) -> BaseDataReader:
        t = turbine.strip().lower()
        if t in {"structure1", "t1", "1"}:
            if not sensor_config:
                raise ValueError("Structure1Reader requires sensor_config.")
            return Structure1Reader(s3_prefix=s3_prefix, sensor_config=sensor_config)

        if t in {"structure3", "structure7", "t3", "t7", "3", "7"}:
            return PeakStructureReader(
                s3_prefix=s3_prefix,
                robust_reading=robust_reading,
                pair_tolerance=pair_tolerance,
                deduplicate_index=deduplicate_index,
                rename_map_c1=rename_map_c1 or {
                    "peak1": "B_flap_nearsided","peak2": "B_edge_nearsided",
                    "peak3": "B_flap_farsided","peak4": "B_edge_farsided",
                    "peak5": "A_flap_nearsided","peak6": "A_edge_nearsided",
                },
                rename_map_c2=rename_map_c2 or {
                    "peak1": "A_flap_farsided","peak2": "A_edge_farsided",
                    "peak3": "C_flap_nearsided","peak4": "C_edge_farsided",
                    "peak5": "C_flap_farsided","peak6": "C_edge_farsided",
                },
                desired_order=desired_order or [
                    "B_flap_nearsided","B_edge_nearsided","B_flap_farsided","B_edge_farsided",
                    "A_flap_nearsided","A_edge_nearsided","A_flap_farsided","A_edge_farsided",
                    "C_flap_nearsided","C_edge_farsided","C_flap_farsided","C_edge_farsided",
                ],
            )
        raise ValueError(f"Unknown turbine '{turbine}'. Expected Structure1, Structure3, or Structure7.")


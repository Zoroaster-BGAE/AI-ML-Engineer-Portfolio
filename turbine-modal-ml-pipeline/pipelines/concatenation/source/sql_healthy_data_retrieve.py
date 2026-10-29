import boto3
import tempfile
import sqlite3
import os
import pandas as pd
from urllib.parse import urlparse
import warnings
from typing import Optional, Sequence, Tuple, Union
from datetime import datetime, time
import sys, os


def print_db_schema_s3(s3_path: str, table_names: list[str] | None = None):
    """
    Fetches an SQLite database from S3 and prints its schema.

    Parameters:
    - s3_path (str): S3 URI ("s3://bucket/key") or "bucket/key" to the .db file.
    - table_names (list[str], optional): List of specific table names to print. 
      If None, prints all tables.
    """
    # parse bucket/key
    parsed = urlparse(s3_path, allow_fragments=False)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    else:
        bucket, key = s3_path.split("/", 1)

    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = tmp.name

    try:
        s3.download_file(bucket, key, temp_path)
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = {name for (name,) in cursor.fetchall()}

        tables_to_print = all_tables if table_names is None else set(table_names) & all_tables

        for tbl in tables_to_print:
            cursor.execute(f"PRAGMA table_info({tbl})")
            cols = ", ".join(f"{name} ({ctype})" for _, name, ctype, *_ in cursor.fetchall())
            print(f"{tbl}: {cols}")

        conn.close()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def print_table_names_s3(s3_path: str):
    """
    Fetches an SQLite database from S3 and prints only the table names.

    Parameters:
    - s3_path (str): S3 URI ("s3://bucket/key") or "bucket/key" to the .db file.
    """
    # parse bucket/key
    parsed = urlparse(s3_path, allow_fragments=False)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    else:
        bucket, key = s3_path.split("/", 1)

    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = tmp.name

    try:
        s3.download_file(bucket, key, temp_path)
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (name,) in cursor.fetchall():
            print(name)

        conn.close()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def load_table_from_s3(
    s3_mdata_path: str,
    table_name: str,
    timestamp_col: Optional[str] = None,
    start_time: str = None,
    end_time: str = None,
    boto_session: Optional[boto3.Session] = None
) -> pd.DataFrame:
    """
    Fetch an SQLite metadata database (.db) from S3 and load a table into a pandas DataFrame,
    optionally filtered by a timestamp column and time range.

    Time strings must be in US format:
    - "YYYY-MM-DD"           → interpreted as the start (00:00:00) or end (23:59:59) of the day
    - "YYYY-MM-DD HH:MM:SS"  → interpreted as an exact timestamp

    Parameters:
    - s3_mdata_path (str): 
        Full S3 path to the SQLite database, either as:
        - "s3://bucket/key.db"
        - "bucket/key.db"
    - table_name (str): 
        Name of the table inside the SQLite database to load.
    - timestamp_col (Optional[str]): 
        Name of the datetime column to filter on. If None, the entire table is returned.
    - start_time (Optional[str]): 
        Lower time bound (inclusive). Ignored if timestamp_col is None.
    - end_time (Optional[str]): 
        Upper time bound (inclusive). Ignored if timestamp_col is None.
    - boto_session (Optional[boto3.Session]): 
        An optional pre-configured boto3 session. If not provided, a default session is used.

    Returns:
    - pd.DataFrame: 
    The resulting table, optionally filtered by timestamp.

    """
    def _parse_us(dt_str: str, default_to_start: bool) -> str:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(dt_str, fmt)
                if fmt == "%Y-%m-%d":
                    dt = datetime.combine(dt.date(), time.min if default_to_start else time.max.replace(microsecond=0))
                elif not default_to_start and dt.time() == time(0, 0, 0):
                    dt = datetime.combine(dt.date(), time.max.replace(microsecond=0))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        raise ValueError(f"Invalid datetime format: {dt_str}")

    # Parse S3 path
    parsed = urlparse(s3_mdata_path, allow_fragments=False)
    if parsed.scheme == "s3":
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
    else:
        bucket, key = s3_mdata_path.split("/", 1)

    # Use provided boto3 session or default
    s3 = (boto_session or boto3.Session()).client("s3")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = tmp.name

    try:
        s3.download_file(bucket, key, temp_path)
        conn = sqlite3.connect(temp_path)

        if not timestamp_col:
            warnings.warn(f"No timestamp_col provided—returning full '{table_name}'", UserWarning)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        else:
            sql = f"SELECT * FROM {table_name}"
            params = {}
            clauses = []

            if start_time:
                params["start"] = _parse_us(start_time, default_to_start=True)
                clauses.append(f"{timestamp_col} >= :start")

            if end_time:
                params["end"] = _parse_us(end_time, default_to_start=False)
                clauses.append(f"{timestamp_col} <= :end")

            if clauses:
                sql += " WHERE " + " AND ".join(clauses)

            df = pd.read_sql_query(sql, conn, params=params)

        conn.close()
        return df

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        
# ==================================================================================================
# Import config for healthy data retrieval, rpm filtering
# ==================================================================================================
sys.path.append('../') # Problematic as if the paths change we might have an issue
from config import (
    S3_MDATA_PATH,
    MAIN_TABLE,
    CORRUPTION_TABLE,
    RPM_TABLE, 
    RPM_RANGE,
    TIMECOL, 
    ABLE_COL, 
    TIMEGAP_COL)
# ==================================================================================================


def fetch_healthy_data(
    s3_mdata_path: str = S3_MDATA_PATH,
    main_table: str = MAIN_TABLE,
    corruption_table: str = CORRUPTION_TABLE,
    timestamp_col: Optional[str] = TIMECOL,
    able_col: str = ABLE_COL,
    timegap_col: str = TIMEGAP_COL,
    start_time: str = None,
    end_time: str = None
) -> pd.DataFrame:
    """
    Returns only the “healthy” rows from `main_table`, defined as:
      1. `able_col`  == True or 1
      2. `timegap_col` == False or 0
      3. AND whose id does not appear as corrupt in `corruption_table`
         (all other columns in that table are 0 or None)

    Steps:
      A) load `main_table` (with optional time slice),
      B) apply (1) and (2),
      C) load `corruption_table` (no timestamp_col),
      D) drop any ids from A) that have ANY non‐zero, non‐null field in C).
    """
    # A) load main table
    df_main = load_table_from_s3(
        s3_mdata_path,
        table_name=main_table,
        timestamp_col=timestamp_col,
        start_time=start_time,
        end_time=end_time
    )

    # B) filter by able_to_open and time_gap
    mask_open    = df_main[able_col].isin([True, 1])
    mask_timegap = df_main[timegap_col].isin([False, 0])
    df_step2     = df_main[mask_open & mask_timegap]

    # C) load corruption_status (no time col)
    df_corr = load_table_from_s3(
        s3_mdata_path,
        table_name=corruption_table,
        timestamp_col=None
    )

    # D) find non‐corrupt ids among those from step2
    corr_flags = df_corr.set_index("id").fillna(0)
    noncorrupt_ids = corr_flags[corr_flags.eq(0).all(axis=1)].index
    valid_ids = set(df_step2["id"]) & set(noncorrupt_ids)

    # final healthy DataFrame
    df_healthy = df_step2[df_step2["id"].isin(valid_ids)].reset_index(drop=True)
    return df_healthy


from config import (
    S3_MDATA_PATH,
    SENSOR_CONFIGURATION,
    RPM_RANGE, 
    RPM_TABLE,
)



def filter_healthy_on_rpm(
    data_file_names: Sequence[str],
    sensor_config: dict[str, list[str]] = SENSOR_CONFIGURATION,
    rpm_range: Union[Tuple[float, float], list[float]] = (9.0, 16.0)
) -> pd.DataFrame:
    """
    Given healthy IDs, compute mean RPM over specified sensors
    and return only those whose RPM lies in rpm_range.

    Parameters
    ----------
    data_file_names
        Iterable of `id` strings.
    sensor_config
        Dict mapping channel names ("ch1", ...) to suffix lists.
    rpm_range
        (min_rpm, max_rpm), both >= 0 and min_rpm <= max_rpm.

    Returns
    -------
    pd.DataFrame
        Columns ['id','rpm'] for ids with rpm in [min_rpm, max_rpm].
    """
    # --- Validate rpm_range ---
    try:
        min_rpm, max_rpm = float(rpm_range[0]), float(rpm_range[1])
    except Exception:
        raise ValueError("rpm_range must be a length-2 sequence of numbers")

    if min_rpm < 0 or max_rpm < 0:
        raise ValueError(f"rpm_range values must be non-negative (got {rpm_range})")
    if min_rpm > max_rpm:
        raise ValueError(f"rpm_range[0] must be <= rpm_range[1] (got {rpm_range})")

    # 1) load the RPM table (no time filter)
    df_rpm = load_table_from_s3(
        S3_MDATA_PATH,
        table_name=RPM_TABLE,
        timestamp_col=None
    )

    # 2) restrict to our healthy ids
    df_rpm = df_rpm[df_rpm['id'].isin(data_file_names)].copy()

    # 3) build list of sensor columns
    sensor_cols = [
        f"{ch}{suffix}"
        for ch, suffixes in sensor_config.items()
        for suffix in suffixes
    ]
    sensor_cols = [c for c in sensor_cols if c in df_rpm.columns]

    # 4) clamp out-of-range readings to NaN, compute mean
    sensor_df = df_rpm[sensor_cols]
    valid_mask = (sensor_df >= 0) & (sensor_df <= 20)
    df_sensors = sensor_df.where(valid_mask, other=pd.NA)

    df_rpm['rpm'] = df_sensors.mean(axis=1, skipna=True)

    # 5) filter by rpm_range
    mask = df_rpm['rpm'].between(min_rpm, max_rpm, inclusive='both')
    return df_rpm.loc[mask, ['id', 'rpm']].reset_index(drop=True)

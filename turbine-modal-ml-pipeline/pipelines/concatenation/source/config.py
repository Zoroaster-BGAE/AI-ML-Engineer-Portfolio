# # config.py
# from dotenv import load_dotenv
# import os
# import json

# # 1)________________________________ DATA READER _____________________________
# # Raw Acceleration Data Reader Configurations

# # # id :
# # STRUCTURE_ID = 'structure1'

# # # This is the folder url of raw data from instalation 'id'. Secrecy should be checked. !
# load_dotenv(dotenv_path=".env")

# # # Here you need to choose which sensors you need for each channel


# STRUCTURE_ID = os.getenv("STRUCTURE_ID", "structure1")
# S3_RDATA_FOLDER_URL = os.getenv("S3_RDATA_FOLDER_URL", f"s3://example-bucket-1/{STRUCTURE_ID}-raw/")

# SENSOR_CONFIGURATION = json.loads(
#     os.getenv("SENSOR_CONFIGURATION",
#               '{"ch1": ["s1", "s2", "s3", "s4", "s5", "s6"], "ch2": ["s1", "s2", "s3", "s4", "s5", "s6"], "ch3": ["s1", "s2", "s3", "s4", "s5", "s6"]}')
# )





# # 2)_____________________________ QUERY HEALTHY DATA __________________________
# # # URL to the Meta Data DB
# #
# S3_MDATA_PATH = os.getenv("S3_MDATA_PATH", "s3://example-bucket-2/metadata.db")

# MAIN_TABLE = os.getenv("MAIN_TABLE", "main_data")
# CORRUPTION_TABLE = os.getenv("CORRUPTION_TABLE", "corruption_status")
# RPM_TABLE = os.getenv("RPM_TABLE", "rpm")
# TIMECOL = os.getenv("TIMECOL", "time")
# FILENAME_COL = os.getenv("FILENAME_COL", "id")
# ABLE_COL = os.getenv("ABLE_COL", "able_to_open")
# TIMEGAP_COL = os.getenv("TIMEGAP_COL", "time_gap")

# RPM_RANGE = tuple(map(int, os.getenv("RPM_RANGE", "9,16").split(",")))
# START_TIME = os.getenv("START_TIME", "2024-01-14")
# END_TIME = os.getenv("END_TIME", "2024-01-15")

# # # =============================================================================
# # # FILTER PARAMETERS




# RPM_BINS = os.getenv("RPM_BINS", "10,16").split(",")
# TOLERANCE = float(os.getenv("TOLERANCE", "0.5"))

# # -------------------------------
# # 4. FILTERING / DESEASONING CONFIG
# # -------------------------------

# CUTOFF = list(map(float, os.getenv("CUTOFF", "0.1,10").split(",")))
# ORDER = int(os.getenv("ORDER", "3"))
# FILTER_SAMPLE_FREQ = int(os.getenv("FILTER_SAMPLE_FREQ", "50"))

# # -------------------------------
# # 5. CONCATENATION CONFIG
# # -------------------------------

# MIN_FILES = int(os.getenv("MIN_FILES", "3"))
# OUTPUT_DIR = os.getenv("OUTPUT_DIR", "s3://example-bucket-1/processed/concatenated")




# <Version 2>
# config.py – concatenation pipeline with clustering‐style validation

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ─── 1) Load .env from project root ────────────────────────────────────────────
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# ─── 2) Date‐range defaults ─────────────────────────────────────────────────────
_min_date     = datetime(2023, 7, 1)        # data‐entry starts 01/07/2025
_max_date_dt  = datetime.now()              # “now” at import time

# ─── 3) Centralized defaults ──────────────────────────────────────────────────
_DEFAULTS = {
    # Data Reader
    "STRUCTURE_ID":         "structure1",
    "S3_RDATA_FOLDER_URL":  f"s3://example-bucket-1/structure1-raw/",
    "SENSOR_CONFIGURATION": {
        "ch1": ["s1","s2","s3","s4","s5","s6"],
        "ch2": ["s1","s2","s3","s4","s5","s6"],
        "ch3": ["s1","s2","s3","s4","s5","s6"],
    },

    # Metadata DB & tables
    "S3_MDATA_PATH":        "s3://example-bucket-2/metadata.db",
    "MAIN_TABLE":           "main_data",
    "CORRUPTION_TABLE":     "corruption_status",
    "RPM_TABLE":            "rpm",
    "TIMECOL":              "time",
    "FILENAME_COL":         "id",
    "ABLE_COL":             "able_to_open",
    "TIMEGAP_COL":          "time_gap",

    # Healthy‐data window (as strings)
    "RPM_RANGE":            (9, 16),
    "START_TIME":           _min_date.strftime("%Y-%m-%d %H:%M:%S"),
    "END_TIME":             _max_date_dt.strftime("%Y-%m-%d %H:%M:%S"),

    # Filter / Deseasoning
    "RPM_BINS":             [10, 16],
    "TOLERANCE":            0.5,
    "CUTOFF":               [0.1, 10.0],
    "ORDER":                3,
    "FILTER_SAMPLE_FREQ":   50,

    # Concatenation
    "MIN_FILES":            3,
    "OUTPUT_DIR":           "s3://example-bucket-1/processed/concatenated",
}

# ─── 4) Helpers ─────────────────────────────────────────────────────────────────
def _get_env(name):
    return os.getenv(name)

def _warn(msg):
    print(f"[CONFIG WARNING] {msg}")

def _validate_type(val_str, cast_fn, default, name):
    try:
        return cast_fn(val_str)
    except Exception:
        _warn(f"{name}='{val_str}' invalid type → default {default!r}")
        return default

def _validate_range(val, check_fn, default, name):
    if not check_fn(val):
        _warn(f"{name}={val!r} out of range → default {default!r}")
        return default
    return val

def _parse_path(name):
    raw = _get_env(name)
    default = _DEFAULTS[name]
    if not raw:
        _warn(f"{name} not set → default {default!r}")
        return default
    if "://" in raw:
        return raw
    try:
        return str(Path(raw).expanduser())
    except Exception:
        _warn(f"{name}='{raw}' invalid path → default {default!r}")
        return default

def _parse_str(name):
    raw = _get_env(name)
    if not raw:
        _warn(f"{name} not set → default {_DEFAULTS[name]!r}")
        return _DEFAULTS[name]
    return raw

def _parse_bool(name):
    raw = _get_env(name)
    default = _DEFAULTS[name]
    if raw is None:
        _warn(f"{name} not set → default {default!r}")
        return default
    r = raw.strip().lower()
    if r in ("true","1","yes","y"):
        return True
    if r in ("false","0","no","n"):
        return False
    _warn(f"{name}='{raw}' invalid boolean → default {default!r}")
    return default

def _parse_json(name, default):
    raw = _get_env(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        _warn(f"{name} JSON parse error → default {default!r}")
        return default

def _parse_date_str(name):
    raw     = _get_env(name)
    default = _DEFAULTS[name]
    if not raw:
        _warn(f"{name} not set → default {default!r}")
        return default

    # try ISO datetime then date-only
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except Exception:
            dt = None
    if dt is None:
        _warn(f"{name}='{raw}' bad format → default {default!r}")
        return default

    # enforce range
    if dt < _min_date or dt > _max_date_dt:
        _warn(
            f"{name}={dt!r} outside [{_min_date!r}, {_max_date_dt!r}] → default {default!r}"
        )
        return default

    # ensure we hand back a full datetime‐string
    if fmt == "%Y-%m-%d":
        dt = dt.replace(hour=0, minute=0, second=0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ─── 5) Parse & Validate ─────────────────────────────────────────────────────────
STRUCTURE_ID         = _parse_str("STRUCTURE_ID")
S3_RDATA_FOLDER_URL  = _parse_path("S3_RDATA_FOLDER_URL")
SENSOR_CONFIGURATION = _validate_range(
    _parse_json("SENSOR_CONFIGURATION", _DEFAULTS["SENSOR_CONFIGURATION"]),
    lambda v: isinstance(v, dict),
    _DEFAULTS["SENSOR_CONFIGURATION"],
    "SENSOR_CONFIGURATION",
)

S3_MDATA_PATH    = _parse_path("S3_MDATA_PATH")
MAIN_TABLE       = _parse_str("MAIN_TABLE")
CORRUPTION_TABLE = _parse_str("CORRUPTION_TABLE")
RPM_TABLE        = _parse_str("RPM_TABLE")
TIMECOL          = _parse_str("TIMECOL")
FILENAME_COL     = _parse_str("FILENAME_COL")
ABLE_COL         = _parse_str("ABLE_COL")
TIMEGAP_COL      = _parse_str("TIMEGAP_COL")

# numeric + range checks
_rpm_range = _get_env("RPM_RANGE") or ",".join(map(str, _DEFAULTS["RPM_RANGE"]))
RPM_RANGE = _validate_range(
    _validate_type(
        _rpm_range,
        lambda s: tuple(int(x) for x in s.split(",")),
        _DEFAULTS["RPM_RANGE"],
        "RPM_RANGE"
    ),
    lambda t: isinstance(t, tuple) and len(t)==2 and t[0] >= 0 and t[1]>=t[0],
    _DEFAULTS["RPM_RANGE"],
    "RPM_RANGE"
)

# date‐strings
START_TIME = _parse_date_str("START_TIME")
END_TIME   = _parse_date_str("END_TIME")

# more numeric
_rpm_bins = _get_env("RPM_BINS") or ",".join(map(str, _DEFAULTS["RPM_BINS"]))
RPM_BINS = _validate_range(
    _validate_type(
        _rpm_bins,
        lambda s: [int(x) for x in s.split(",")],
        _DEFAULTS["RPM_BINS"],
        "RPM_BINS"
    ),
    lambda L: all(isinstance(i,int) and i>=0 for i in L),
    _DEFAULTS["RPM_BINS"],
    "RPM_BINS"
)

TOLERANCE          = _validate_range(
    _validate_type(_get_env("TOLERANCE"), float, _DEFAULTS["TOLERANCE"], "TOLERANCE"),
    lambda v: v >= 0.0,
    _DEFAULTS["TOLERANCE"],
    "TOLERANCE"
)
CUTOFF             = _validate_range(
    _validate_type(_get_env("CUTOFF"), lambda s: [float(x) for x in s.split(",")], _DEFAULTS["CUTOFF"], "CUTOFF"),
    lambda L: len(L)==2 and L[0]>=0 and L[1]>=L[0],
    _DEFAULTS["CUTOFF"],
    "CUTOFF"
)
ORDER              = _validate_range(
    _validate_type(_get_env("ORDER"), int, _DEFAULTS["ORDER"], "ORDER"),
    lambda v: v >= 1,
    _DEFAULTS["ORDER"],
    "ORDER"
)
FILTER_SAMPLE_FREQ = _validate_range(
    _validate_type(_get_env("FILTER_SAMPLE_FREQ"), int, _DEFAULTS["FILTER_SAMPLE_FREQ"], "FILTER_SAMPLE_FREQ"),
    lambda v: v >= 1,
    _DEFAULTS["FILTER_SAMPLE_FREQ"],
    "FILTER_SAMPLE_FREQ"
)

MIN_FILES   = _validate_range(
    _validate_type(_get_env("MIN_FILES"), int, _DEFAULTS["MIN_FILES"], "MIN_FILES"),
    lambda v: v >= 1,
    _DEFAULTS["MIN_FILES"],
    "MIN_FILES"
)
OUTPUT_DIR  = _parse_path("OUTPUT_DIR")


# ─── 6) Summary log ─────────────────────────────────────────────────────────────
print(f"[CONFIG] STRUCTURE_ID={STRUCTURE_ID}")
print(f"[CONFIG] S3_RDATA_FOLDER_URL={S3_RDATA_FOLDER_URL}")
print(f"[CONFIG] SENSOR_CONFIGURATION={SENSOR_CONFIGURATION}")
print(f"[CONFIG] S3_MDATA_PATH={S3_MDATA_PATH}")
print(f"[CONFIG] MAIN_TABLE={MAIN_TABLE}")
print(f"[CONFIG] CORRUPTION_TABLE={CORRUPTION_TABLE}")
print(f"[CONFIG] RPM_TABLE={RPM_TABLE}")
print(f"[CONFIG] TIMECOL={TIMECOL}")
print(f"[CONFIG] FILENAME_COL={FILENAME_COL}")
print(f"[CONFIG] ABLE_COL={ABLE_COL}")
print(f"[CONFIG] TIMEGAP_COL={TIMEGAP_COL}")
print(f"[CONFIG] RPM_RANGE={RPM_RANGE}")
print(f"[CONFIG] START_TIME={START_TIME}")
print(f"[CONFIG] END_TIME={END_TIME}")
print(f"[CONFIG] RPM_BINS={RPM_BINS}")
print(f"[CONFIG] TOLERANCE={TOLERANCE}")
print(f"[CONFIG] CUTOFF={CUTOFF}")
print(f"[CONFIG] ORDER={ORDER}")
print(f"[CONFIG] FILTER_SAMPLE_FREQ={FILTER_SAMPLE_FREQ}")
print(f"[CONFIG] MIN_FILES={MIN_FILES}")
print(f"[CONFIG] OUTPUT_DIR={OUTPUT_DIR}")

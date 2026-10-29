import smart_open
import json
import pandas as pd
from datetime import datetime
import os
import boto3
from dotenv import load_dotenv

from config import (S3_RDATA_FOLDER_URL, 
                    SENSOR_CONFIGURATION, 
                    RPM_TABLE, 
                    FILENAME_COL)

def connect_dotenv_aws(secrets_path="../.env.secrets"):
    """
    Load AWS creds from a .env file into the environment, validate them,
    and return a boto3 Session.
    """
    
    load_dotenv(dotenv_path=secrets_path, override=False)

    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")

    if not all([aws_key, aws_secret, aws_region]):
        missing = [name for name, val in [
            ("AWS_ACCESS_KEY_ID", aws_key),
            ("AWS_SECRET_ACCESS_KEY", aws_secret),
            ("AWS_DEFAULT_REGION", aws_region),
        ] if not val]
        raise RuntimeError(f"Missing AWS credentials: {', '.join(missing)}")

    return boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region
    )

def read_data_minimal(file_name, 
                      s3_rdata_folder_url = S3_RDATA_FOLDER_URL, 
                      sensor_config = SENSOR_CONFIGURATION):
    """
    Generic data reader for COMPANY sensor .gz JSON files from S3.

    Parameters
    ----------
    file_name : str
        Filename of the .gz file to read.
    s3_folder_url : str
        Base URL of the S3 folder (e.g., 's3://bucket/folder/').
    sensor_config : dict
        Dictionary mapping channel names (e.g., 'ch1') to sensor names (e.g., ['s1', 's2', ...]).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'time' and sensor data as specified in the config.
    """
    
    # Build S3 path
    url = f"{s3_rdata_folder_url}{file_name}"

    # Read JSON data from .gz
    with smart_open.open(url, 'r') as fin:
        data = json.loads(fin.read())

    # Create base DataFrame
    df = pd.DataFrame({
        'time': data['time'],
        'channel': data['channel'],
        'signals': data['signals']
    })

    # Unpack the sensor signal lists into separate columns
    signal_df = pd.DataFrame(df['signals'].tolist(), index=df.index)
    
    # Temporary sensor labels, assuming max 9 signals as in original code
    signal_df.columns = [f's{i+1}' for i in range(signal_df.shape[1])]
    
    # Merge with base
    df = pd.concat([df[['time', 'channel']], signal_df], axis=1)

    # Assemble final output
    final_df = pd.DataFrame()
    
    # Use the time values from the first available channel
    first_channel_key = list(sensor_config.keys())[0]
    first_channel_num = int(first_channel_key.replace('ch', ''))
    first_df = df[df['channel'] == first_channel_num]
    final_df['time'] = first_df['time'].values

    # For each channel in config
    for ch_key, sensors in sensor_config.items():
        ch_num = int(ch_key.replace('ch', ''))
        ch_df = df[df['channel'] == ch_num]

        for sensor in sensors:
            final_df[f'{ch_key}{sensor}'] = ch_df[sensor].values

    return final_df


#---------------------------------------------------------------------------------------------
def date_extractor(fname):
    # e.g. "Structure1_data_20230822102042"
    ts_str = fname.split('_')[-1]             # "20230822102042"
    return datetime.strptime(ts_str, "%Y%m%d%H%M%S")


def create_daily_dict(df, bins, tol, date_extractor = date_extractor, 
                      filename_col = FILENAME_COL, 
                      rpm_col = RPM_TABLE):
    """
    Builds a nested dict of { YYYYMMDD: { bin_center: { filename: rpm, ... }, ... }, ... }
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least `filename_col` and `rpm_col`.
    bins : iterable of float
        The centers of the RPM bins to look for.
    tol : float
        Half‐width of each bin (so bin_center ± tol).
    date_extractor : callable
        A function `date_extractor(filename: str) -> datetime.datetime`
        that pulls the timestamp out of your filename string.
    filename_col : str, default 'id'
        Column in `df` containing the filenames/IDs.
    rpm_col : str, default 'rpm'
        Column in `df` containing the RPM values.
    
    Returns
    -------
    dict of dicts
        Mapping date‐string (YYYYMMDD) → bin_center (as str) → { filename: rpm, ... }
    """
    # 1) Extract datetime from filename
    df = df.copy()
    df['time'] = df[filename_col].apply(date_extractor)
    
    # 2) Pull out just the date as YYYYMMDD
    df['date'] = df['time'].dt.strftime('%Y%m%d')
    
    # 3) Normalize column names (so we can write 'RPM' in the filters below)
    df.rename(columns={rpm_col: 'RPM'}, inplace=True)
    
    daily_dict = {}
    
    # 4) Group by date, then by each bin
    for day, day_group in df.groupby('date'):
        daily_dict[day] = {}
        for bin_center in bins:
            low, high = float(bin_center) - tol, float(bin_center) + tol
            mask = (day_group['RPM'] >= low) & (day_group['RPM'] <= high)
            bin_group = day_group.loc[mask]
            if not bin_group.empty:
                # Use stringified bin_center as the key
                daily_dict[day][str(bin_center)] = dict(
                    zip(bin_group[filename_col], bin_group['RPM'])
                )
    return daily_dict
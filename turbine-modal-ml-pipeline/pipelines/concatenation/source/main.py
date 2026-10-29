#!/usr/bin/env python3
"""
main.py

Script to run the data processing pipeline:
1. Fetch healthy data from S3/DB
2. Filter by RPM
3. Create daily dictionary of files
4. Process and analyze daily data
"""
import os
import logging
import argparse

# Project imports
from sql_healthy_data_retrieve import fetch_healthy_data, filter_healthy_on_rpm
from utils import connect_dotenv_aws, create_daily_dict, read_data_minimal

from conc_operations import process_daily_dict
from config import (
    START_TIME,
    END_TIME,
    RPM_RANGE,
    RPM_BINS,
    TOLERANCE,
    SENSOR_CONFIGURATION,
    CUTOFF,
    FILTER_SAMPLE_FREQ,
    ORDER,
    MIN_FILES,
    OUTPUT_DIR,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process healthy data and generate daily metrics."
    )
    parser.add_argument(
        "--start-time",
        default=START_TIME,
        help="Start time for data fetch (e.g. '2025-01-01T00:00:00Z')."
    )
    parser.add_argument(
        "--end-time",
        default=END_TIME,
        help="End time for data fetch (e.g. '2025-01-07T00:00:00Z')."
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory to save output files."
    )
    parser.add_argument(
        "--rpm-range",
        default=None,
        help="RPM range for filtering healthy data (overrides config)."
    )
    parser.add_argument(
        "--rpm-bins",
        default=None,
        help="RPM bins for grouping (overrides config)."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Tolerance for RPM binning (overrides config)."
    )
    parser.add_argument(
        "--sensor-configuration",
        default=None,
        help="Sensor configuration settings (overrides config)."
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Cutoff threshold for processing (overrides config)."
    )
    parser.add_argument(
        "--filter-sample-freq",
        type=float,
        default=None,
        help="Filter sample frequency (overrides config)."
    )
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="Order parameter for processing (overrides config)."
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=None,
        help="Minimum number of files for processing (overrides config)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
     # Determine whether we're writing locally or to S3
    is_s3 = bool(args.output_dir and args.output_dir.startswith("s3://"))
    if not is_s3 and args.output_dir:
        # local path → make sure it exists
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info("Starting data processing pipeline.")

    # Use overrides if provided, else fall back to config
    start_time = args.start_time
    end_time = args.end_time
    rpm_range = args.rpm_range or RPM_RANGE
    rpm_bins = args.rpm_bins or RPM_BINS
    tolerance = args.tolerance or TOLERANCE
    sensor_configuration = args.sensor_configuration or SENSOR_CONFIGURATION
    cutoff = args.cutoff or CUTOFF
    filter_sample_freq = args.filter_sample_freq or FILTER_SAMPLE_FREQ
    order = args.order or ORDER
    min_files = args.min_files or MIN_FILES

    # Step 0: connent to s3
    session = connect_dotenv_aws() 

    # Step 1: Fetch healthy data
    logging.info(f"Fetching healthy data from {start_time} to {end_time}.")
    h_df = fetch_healthy_data(start_time=start_time, end_time=end_time)

    # Step 2: Filter by RPM
    logging.info(f"Filtering data by RPM range: {rpm_range}.")
    fh_df = filter_healthy_on_rpm(h_df['id'], rpm_range=rpm_range)

    # Step 3: Create daily dictionary
    logging.info("Creating daily dictionary.")
    daily_dict = create_daily_dict(fh_df, bins=rpm_bins, tol=tolerance)

    # Step 4: Process daily dictionary
    logging.info("Processing daily dictionary.")
    pdaily_dict = process_daily_dict(
        daily_dict,
        output_dir=args.output_dir,
        MIN_FILES=min_files,
        CUTOFF=cutoff,
        FILTER_SAMPLE_FREQ=filter_sample_freq,
        ORDER=order,
        sensor_configuration=sensor_configuration,
    )

    logging.info(f"Data processing complete. Results saved in {args.output_dir}.")

if __name__ == "__main__":
    main()

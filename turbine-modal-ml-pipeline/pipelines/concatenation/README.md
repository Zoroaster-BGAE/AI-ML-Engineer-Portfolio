README

Project: Concatenation Data Processing Pipeline
Overview
This pipeline retrieves raw sensor data from S3, filters out unhealthy or out‑of‑range files, concatenates healthy files by RPM bins for each day, and writes the results back to S3. The workflow consists of:

1. Metadata Fetch: Download metadata tables (main, rpm, and corruption_status) from the configured database/S3.

2. RPM Filtering: Using RPM_RANGE, select only files within the desired RPM interval.

3. Daily Loop: For each day between START_TIME and END_TIME:

   * List candidate files in that day’s folder.

   * For each RPM bin in RPM_BINS, download and inspect files until MIN_FILES healthy files are collected.

   * Concatenate the healthy files (in time order) into a single object.

   * Upload the result to OUTPUT_DIR as {YYYYMMDD}_{RPM_bin}_conc.pkl.

      4. Finalization: After processing all days and bins, upload a summary final_daily_dict.pkl and exit.

________________


Prerequisites
Before running the pipeline, ensure you have the following installed and configured:
         * Docker ≥ 20.x: To build and run the containerized application.

         * AWS CLI (optional, for manual testing): Configure with aws configure to verify S3_RDATA_FOLDER_URL and S3_MDATA_PATH access.

         * Python 3.11 (for local development): If running scripts directly, install via python.org.

         * Credentials:

            * An AWS IAM user with permission to read from your raw data bucket and write processed results to your output bucket.

            * Store these in .env.secrets and .env files (see Configuration below).

________________


Configuration
The pipeline is fully configured via environment variables. You can supply them in two ways:
               1. Dotenv files:

                  * .env.secrets: Sensitive values (AWS access key and secret).

                  * .env: Non‑sensitive defaults and overrides.

                     2. Override on the command line using -e KEY=VALUE flags.

Environment Files Example
# .env.secrets
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
AWS_DEFAULT_REGION=eu-central-1


# .env
STRUCTURE_ID=structure1
S3_RDATA_FOLDER_URL=s3://example-bucket-1/structure1-raw/
S3_MDATA_PATH=s3://example-bucket-2/metadata.db
MAIN_TABLE=main_data
CORRUPTION_TABLE=corruption_status
RPM_TABLE=rpm
TIMECOL=time
FILENAME_COL=id
ABLE_COL=able_to_open
TIMEGAP_COL=time_gap
RPM_RANGE=9,16
RPM_BINS=10,16
TOLERANCE=0.5
CUTOFF=0.1,10.0
ORDER=3
FILTER_SAMPLE_FREQ=50
MIN_FILES=3
OUTPUT_DIR=s3://example-bucket-1/processed/concatenated


Configuration Parameters
Variable
	Type
	Description
	Default
	Format / Allowed Values
	STRUCTURE_ID
	string
	Identifier for this sensor group.
	—
	alphanumeric
	S3_RDATA_FOLDER_URL
	URI
	Root S3 URI for raw data.
	—
	s3://bucket/path/
	S3_MDATA_PATH
	URI
	Path to metadata DB file.
	—
	s3://... or /path/to/db
	SENSOR_CONFIGURATION
	JSON
	Mapping of channels to sensor IDs.
	—
	JSON object
	MAIN_TABLE
	string
	Table name for file records.
	main_data
	SQL table name
	CORRUPTION_TABLE
	string
	Table name for corruption flags.
	corruption_status
	SQL table name
	RPM_TABLE
	string
	Table name for RPM measurements.
	rpm
	SQL table name
	TIMECOL
	string
	Column name for timestamps.
	time
	column name
	FILENAME_COL
	string
	Column name for file identifiers.
	id
	column name
	ABLE_COL
	string
	Column name for openable flag.
	able_to_open
	column name
	TIMEGAP_COL
	string
	Column name for time gap metric.
	time_gap
	column name
	RPM_RANGE
	pair
	Global RPM filter range (min,max).
	(0,∞)
	min,max
	RPM_BINS
	list
	Edges for RPM grouping.
	[min,max]
	val1,val2,...
	TOLERANCE
	float
	Relative tolerance on RPM.
	0.0
	≥ 0.0
	CUTOFF
	pair
	Signal amplitude thresholds (min,max).
	(0.0,∞)
	min,max
	ORDER
	int
	Filter order for time-series processing.
	2
	positive integer
	FILTER_SAMPLE_FREQ
	float
	Downsampling frequency (Hz).
	metadata default
	> 0
	MIN_FILES
	int
	Minimum healthy files per bin before concatenation.
	1
	≥1
	START_TIME
	ISO8601
	Inclusive fetch start.
	—
	YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD
	END_TIME
	ISO8601
	Exclusive fetch end.
	—
	YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD
	OUTPUT_DIR
	URI
	S3 URI or local path for results.
	—
	s3://... or /path/to/output
	________________


Scripts
The following scripts form the core of the application. Each plays a specific role in the end‑to‑end data processing pipeline:
                        * main.py: Entry point for the pipeline. Loads configuration, fetches metadata, constructs the daily file dictionary, and delegates to processing functions.

                        * config.py: Centralizes environment variable parsing, type/range validation, and default management; prints a summary of all runtime parameters.

                        * utils.py: Helper module that establishes AWS sessions, reads and parses compressed JSON sensor files into DataFrames, and constructs the daily_dict mapping dates and RPM bins to file lists.

                        * sql_healthy_data_retrieve.py: Retrieves metadata tables (main, rpm, corruption_status) via SQL/S3, filters out corrupted or timestamp‑out‑of‑range records, and returns a consolidated DataFrame of healthy file entries.

                        * conc_operations.py: Implements core data operations: deseasonalization (sinusoidal fitting + bandpass filtering), per‑bin concatenation logic, and saving pickled outputs (locally or to S3).

Installation
Clone the repo:

git clone https://github.com/your-org/concat-app.git
cd concat-app
                           1. Build the Docker image:



                              2.  docker build -t company/concat-app:1.2 .


Usage
Run the container interactively to stream logs:
docker run -it \
  --env-file .env.secrets \
  --env-file .env \
  -e START_TIME="2024-07-01T00:00:00Z" \
  -e END_TIME="2024-07-03T00:00:00Z" \
  company/concat-app:1.2


Or detach and follow logs:
cid=$(docker run -d --env-file .env.secrets --env-file .env -e START_TIME=2024-07-01T00:00:00Z -e END_TIME=2024-07-03T00:00:00Z company/concat-app:1.2)
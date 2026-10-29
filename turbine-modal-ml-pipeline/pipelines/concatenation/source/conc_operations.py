import numpy as np
import pandas as pd
import time
import gc
import logging
import os
import pickle
import tempfile
import boto3

from smart_open import open as s3open
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pathlib import Path

from utils import read_data_minimal

def butter_lowpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype='bandpass', analog=False) # Maybe reduntant function

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data) # Maybe use signal.filtfilt as it is done in PyOMA
    return y

def bandpass_filter(dataframe, cutoff, fs, order):
    return dataframe.transform(lambda x: butter_lowpass_filter(x, cutoff, fs, order))

def sinusoidal(time, frequency, amplitude, phase,  k1, k2, k3):
    '''
    sinusoidal function
    
    Parameters
    ----------
    time: float
    frequency: float
    amplitude: float
    phase: float
    k1: float
    k2: float
    k3: float
    
    Returns
    ----------
    function output : array
    '''
    return amplitude * np.sin(2*np.pi*frequency * time + phase)+k1*time**2+k2*time+k3

def sinfit(sensor): 
    '''required parameters: dataframe with time and column(s) with sensor data to be fitted.

       Parameters: sensor
       Returns: parameters of sinoid function, y values of fitted sinoid function.
    '''
    y=[] # to save predicted values  
    l=len(sensor.columns) # number of columns
    r = {} #defines r squared value list (goodness of fit)
    Param_dict={} # dict to save parameters of predicted sinusoidal fit
    time=sensor.time-sensor.time.min()
    for i in range(1,l): # to loop sensor columns one-by-one
        # time=sensor.time-sensor.time.min() # adjusting time values from 0 to 35 seconds 
        sensor=sensor.iloc[:,i] # selecting sensor values
        # ma=sensor.rolling(int(len(sensor))).mean()
        xy=pd.concat({'time': time, 'ac': sensor}) # dataframe with adjuested time and sensor values
        tt=time.values # time array
        yy=sensor.values # sensor value array
        
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0])) 
        Fyy = abs(np.fft.fft(yy))
        freq0=abs(ff[np.argmax(Fyy[1:])+1]) # first guess of frequency
        
        xdata=xy.dropna().time # time axis
        ydata=xy.dropna().ac # sensor values axis
        
        # amp0=ydata.max() # first guess of amplitude
        if yy[10]>=0:
            amp0=yy.max()
        elif yy[10]<0:
            amp0=yy.min()
        phase0=0 # first guess of phase
        k1=0 # first guess of constant
        k2=0
        k3=0
        '''
        predicting best possible parameters for given function and the data. Algorithm starts
        from given first guess, parameter values.
        
        '''
        try:
            popt, pcov = curve_fit(sinusoidal, xdata, ydata, p0=(freq0, amp0, phase0, k1, k2, k3))
            y_pred = sinusoidal(time, *popt)
        except RuntimeError:
            popt=np.empty((6))
            popt[:] = np.nan
            y_pred = sinusoidal(time, *popt)
        # predicted values using predicted parameters
        # y_pred = sinusoidal(time, *popt)
        
        f=popt[0] # predicted frequency of the sensor
        key=sensor.name
        # evaluating the fit using R2 score
        try:
            r[key]=r2_score(sensor, y_pred)
        except ValueError:
            r[key]=np.nan
        # print(f'estimated t=frequency is {f} for {key} and curve is fitted with r2 score {r[key]}')
        y_pred.rename(key, inplace=True)
        # print(f'estimated t=frequency is {f} for {key} and curve is fitted with r2 score {r}')
        # print(f'  parameters of curve: {popt}')
        y.append(y_pred)
        # plt.plot(time, y_pred)
        Param_dict={**Param_dict, key: popt} # parameter dict which will be returned 
    yfitdf=pd.concat([time]+ y, axis=1) # to return predicted values
    return Param_dict, yfitdf, r


def deseasonal_simple(expanded_df, cutoff, fs, order):
    """
    Given an expanded df with synthetic columns,
    returns a DataFrame of deseasonalized, filtered data
    for raw sensors and edge-wise / flap-wise average sensors.
    
    Parameters
    ----------
    expanded_df : pd.DataFrame
        DataFrame containing at least 19 columns
        (time + 18 sensor columns).
    cutoff : list of float
        Two-element list [low_freq, high_freq] for bandpass.
    fs : float
        Sampling frequency.
    order : int
        Order of the filter.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with the same columns as the first 19 columns of
        `expanded_df`, except the sensor data has been deseasonalized
        and then bandpass-filtered.
    """
    
    # 1. Extract first 19 columns (time + 18 sensors)
    df = expanded_df.iloc[:, :19]
    
    # 2. Extract time
    time_series = df['time']
    
    # 3. Build a dictionary of fitted values
    fit_dict = {'time': time_series}
    
    # Here we assume sinfit is available in your environment
    # sinfit(df_col) -> (params, fitted_values, r2)
    
    # 4. Sinusoidal fitting for each sensor column
    for col in df.columns:
        if col == 'time':
            continue  # Skip the time column
        sensor = pd.DataFrame({'time': time_series, col: df[col]})
        
        # Fit a sine wave model
        params, fit_values, r2 = sinfit(sensor)
        
        # Store the fitted model's output
        fit_dict[col] = fit_values[col]
    
    # Convert fitted results to a DataFrame
    fit_df = pd.DataFrame(fit_dict)
    
    # 5. Subtract the fit from the raw data to get deseasonalized data
    #    (skip 'time' column in the subtraction)
    deseasonal_df = df.iloc[:, 1:] - fit_df.iloc[:, 1:]
    
    # Here we assume bandpass_filter is available in your environment
    # bandpass_filter(data, cutoff, fs, order) -> returns filtered data
    
    # 6. Apply a bandpass filter to the deseasonalized columns
    filter_df = bandpass_filter(deseasonal_df, cutoff, fs, order)
    
    # 7. Reattach the time column to the filtered data
    filter_df['time'] = time_series
    
    # Reorder if you want 'time' to be the first column
    cols = ['time'] + [c for c in filter_df.columns if c != 'time']
    filter_df = filter_df[cols]
    
    # 8. Return the final DataFrame
    return filter_df


def process_daily_dict(
    daily_dict,
    output_dir=None,
    MIN_FILES=5,
    CUTOFF=None,
    FILTER_SAMPLE_FREQ=None,
    ORDER=None,
    sensor_configuration=None
):
    # Logger setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    is_s3 = bool(output_dir and output_dir.startswith('s3://'))
    if output_dir and not is_s3:
        outdir_path = Path(output_dir)  # <-- use Path
        outdir_path.mkdir(parents=True, exist_ok=True)
        local_log = outdir_path / 'process.log'
    else:
        local_log = Path(tempfile.gettempdir()) / 'process.log'

    fh = logging.FileHandler(str(local_log))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    if not output_dir:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(sh)

    # Default sensor configuration
    if sensor_configuration is None:
        sensor_configuration = {
            'ch1': [f's{i}' for i in range(1,7)],
            'ch2': [f's{i}' for i in range(1,7)],
            'ch3': [f's{i}' for i in range(1,7)],
        }
    COLUMNS_CHOICE = [f"{ch}{sensor}" for ch, sensors in sensor_configuration.items() for sensor in sensors]

    start_time = time.time()
    total_files = sum(
        len(files_dict)
        for rpm_bin_dict in daily_dict.values()
        for files_dict in rpm_bin_dict.values()
    )
    processed_files = 0
    daily_dict_conc = {}

    # Main loop
    for day, rpm_bin_dict in daily_dict.items():
        logger.info(f"Processing day: {day} | Elapsed: {time.time() - start_time:.2f}s")
        daily_dict_conc[day] = {}
        for rpm_bin, files_dict in rpm_bin_dict.items():
            logger.info(f"  RPM bin: {rpm_bin} | Elapsed: {time.time() - start_time:.2f}s")

            rpm_vals = list(files_dict.values())
            mean_rpm = sum(rpm_vals) / len(rpm_vals) if rpm_vals else 0
            sorted_files = sorted(files_dict.items(), key=lambda x: abs(x[1] - mean_rpm))

            df_list = []
            for filename, rpm_val in sorted_files:
                if len(df_list) >= MIN_FILES:
                    break
                processed_files += 1
                gz = filename + '.gz'
                logger.info(
                    f"Checking {processed_files}/{total_files}: {gz} (RPM={rpm_val:.3f}, mean={mean_rpm:.3f}) | "
                    f"Elapsed: {time.time() - start_time:.2f}s"
                )
                try:
                    df = read_data_minimal(gz)
                    df = deseasonal_simple(df, CUTOFF, FILTER_SAMPLE_FREQ, ORDER)
                    df = df[COLUMNS_CHOICE]
                except Exception as e:
                    logger.error(f"Error processing {gz}: {e} | Elapsed: {time.time() - start_time:.2f}s")
                    continue
                if df.isna().any().any():
                    logger.info(f"NaNs in {gz}, skipping | Elapsed: {time.time() - start_time:.2f}s")
                    continue
                df_list.append(df)
                logger.info(f"{gz} passed ({len(df_list)}/{MIN_FILES}) | Elapsed: {time.time() - start_time:.2f}s")

            # Concatenate and save
            if len(df_list) >= MIN_FILES:
                logger.info(f"--> {rpm_bin} has {len(df_list)} files >= {MIN_FILES} | Elapsed: {time.time() - start_time:.2f}s")
                conc = pd.concat(df_list[:MIN_FILES], ignore_index=True)
                key = f"{day}_{rpm_bin}_conc.pkl"

                if output_dir:
                    if is_s3:
                        bucket, prefix = output_dir[5:].split('/', 1)
                        obj_key = f"{prefix}/{key}"
                        tmp_file = tempfile.NamedTemporaryFile(delete=False)
                        try:
                            pd.to_pickle(conc, tmp_file.name)
                            tmp_file.close()
                            s3 = boto3.client('s3')
                            s3.upload_file(tmp_file.name, bucket, obj_key)
                            logger.info(f"Saved conc to s3://{bucket}/{obj_key} | Elapsed: {time.time() - start_time:.2f}s")
                        finally:
                            os.unlink(tmp_file.name)
                    else:
                        # --- FIXED: build a clean local path without .resolve() on strings
                        conc_path = Path(output_dir) / key
                        conc.to_pickle(conc_path)
                        logger.info(f"Saved conc to {conc_path} | Elapsed: {time.time() - start_time:.2f}s")

                daily_dict_conc[day][rpm_bin] = conc
                del conc
                gc.collect()
            else:
                logger.info(f"--> {rpm_bin} only {len(df_list)} files < {MIN_FILES}, skipping | Elapsed: {time.time() - start_time:.2f}s")
            del df_list
            gc.collect()

    # Deduplicate
    for day in daily_dict_conc:
        for rpm_bin in daily_dict_conc[day]:
            daily_dict_conc[day][rpm_bin] = daily_dict_conc[day][rpm_bin].drop_duplicates()

    # Save full dictionary
    if output_dir:
        key_all = 'final_daily_dict.pkl'
        if is_s3:
            bucket, prefix = output_dir[5:].split('/', 1)
            obj_key = f"{prefix}/{key_all}"
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            try:
                pd.to_pickle(daily_dict_conc, tmp_file.name)
                tmp_file.close()
                s3 = boto3.client('s3')
                s3.upload_file(tmp_file.name, bucket, obj_key)
                logger.info(f"Saved full dict to s3://{bucket}/{obj_key} | Elapsed: {time.time() - start_time:.2f}s")
            finally:
                os.unlink(tmp_file.name)
        else:
            # --- FIXED: clean local path construction
            dict_path = Path(output_dir) / key_all
            pd.to_pickle(daily_dict_conc, dict_path)
            logger.info(f"Saved full dict to {dict_path} | Elapsed: {time.time() - start_time:.2f}s")

    # Upload log if using S3
    if output_dir and is_s3:
        bucket, prefix = output_dir[5:].split('/', 1)
        s3 = boto3.client('s3')
        s3.upload_file(str(local_log), bucket, f"{prefix}/process.log")

    logger.info(f"All done! Processed {processed_files}/{total_files} files in {time.time() - start_time:.2f}s")
    return daily_dict_conc



def process_daily_dict_(
    daily_dict,
    output_dir=None,
    MIN_FILES=5,
    CUTOFF=None,
    FILTER_SAMPLE_FREQ=None,
    ORDER=None,
    sensor_configuration=None
):
    """
    Processes daily_dict:
      1. For each day -> rpm_bin -> list of files,
         - compute mean rpm,
         - sort files by distance from mean,
         - read+process up to MIN_FILES valid dataframes,
         - concatenate them, 
         - save the result to PKL in S3 or locally.
      2. Return a dictionary of all final concatenated DataFrames,
         and optionally save the *entire dictionary* to PKL in S3 or locally.

    Arguments:
        daily_dict: dict of day -> rpm_bin -> {filename: rpm_val}
        output_dir: str or None, S3 prefix (s3://bucket/prefix) or local dir
        MIN_FILES: int, max number of valid files per bin
        CUTOFF: float, cutoff frequency for deseasonal_simple
        FILTER_SAMPLE_FREQ: float, sample frequency for deseasonal_simple
        ORDER: int, filter order for deseasonal_simple
        sensor_configuration: dict mapping channel keys to list of sensor suffixes
    """
    

    # Logger setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    is_s3 = bool(output_dir and output_dir.startswith('s3://'))
    if output_dir and not is_s3:
        os.makedirs(output_dir, exist_ok=True)
        local_log = os.path.join(output_dir, 'process.log')
    else:
        local_log = os.path.join(tempfile.gettempdir(), 'process.log')

    fh = logging.FileHandler(local_log)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    if not output_dir:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(sh)

    # Default sensor configuration
    if sensor_configuration is None:
        sensor_configuration = {
            'ch1': [f's{i}' for i in range(1,7)],
            'ch2': [f's{i}' for i in range(1,7)],
            'ch3': [f's{i}' for i in range(1,7)],
        }
    COLUMNS_CHOICE = [f"{ch}{sensor}" for ch, sensors in sensor_configuration.items() for sensor in sensors]

    start_time = time.time()
    total_files = sum(
        len(files_dict)
        for rpm_bin_dict in daily_dict.values()
        for files_dict in rpm_bin_dict.values()
    )
    processed_files = 0
    daily_dict_conc = {}

    # Main loop
    for day, rpm_bin_dict in daily_dict.items():
        logger.info(f"Processing day: {day} | Elapsed: {time.time() - start_time:.2f}s")
        daily_dict_conc[day] = {}
        for rpm_bin, files_dict in rpm_bin_dict.items():
            logger.info(f"  RPM bin: {rpm_bin} | Elapsed: {time.time() - start_time:.2f}s")

            # Compute mean RPM and sort files
            rpm_vals = list(files_dict.values())
            mean_rpm = sum(rpm_vals) / len(rpm_vals) if rpm_vals else 0
            sorted_files = sorted(files_dict.items(), key=lambda x: abs(x[1] - mean_rpm))

            df_list = []
            for filename, rpm_val in sorted_files:
                if len(df_list) >= MIN_FILES:
                    break
                processed_files += 1
                gz = filename + '.gz'
                logger.info(
                    f"Checking {processed_files}/{total_files}: {gz} (RPM={rpm_val:.3f}, mean={mean_rpm:.3f}) | "
                    f"Elapsed: {time.time() - start_time:.2f}s"
                )
                try:
                    df = read_data_minimal(gz)
                    df = deseasonal_simple(df, CUTOFF, FILTER_SAMPLE_FREQ, ORDER)
                    df = df[COLUMNS_CHOICE]
                except Exception as e:
                    logger.error(f"Error processing {gz}: {e} | Elapsed: {time.time() - start_time:.2f}s")
                    continue
                if df.isna().any().any():
                    logger.info(f"NaNs in {gz}, skipping | Elapsed: {time.time() - start_time:.2f}s")
                    continue
                df_list.append(df)
                logger.info(f"{gz} passed ({len(df_list)}/{MIN_FILES}) | Elapsed: {time.time() - start_time:.2f}s")

            # Concatenate and save
            if len(df_list) >= MIN_FILES:
                logger.info(f"--> {rpm_bin} has {len(df_list)} files >= {MIN_FILES} | Elapsed: {time.time() - start_time:.2f}s")
                conc = pd.concat(df_list[:MIN_FILES], ignore_index=True)
                key = f"{day}_{rpm_bin}_conc.pkl"
                if output_dir:
                    if is_s3:
                        bucket, prefix = output_dir[5:].split('/', 1)
                        obj_key = f"{prefix}/{key}"
                        tmp_file = tempfile.NamedTemporaryFile(delete=False)
                        try:
                            pd.to_pickle(conc, tmp_file.name)
                            tmp_file.close()
                            s3 = boto3.client('s3')
                            s3.upload_file(tmp_file.name, bucket, obj_key)
                            logger.info(f"Saved conc to s3://{bucket}/{obj_key} | Elapsed: {time.time() - start_time:.2f}s")
                        finally:
                            os.unlink(tmp_file.name)
                    else:
                        path = os.path.join(output_dir, key).resolve()
                        conc.to_pickle(path)
                        logger.info(f"Saved conc to {path} | Elapsed: {time.time() - start_time:.2f}s")
                daily_dict_conc[day][rpm_bin] = conc
                del conc
                gc.collect()
            else:
                logger.info(f"--> {rpm_bin} only {len(df_list)} files < {MIN_FILES}, skipping | Elapsed: {time.time() - start_time:.2f}s")
            del df_list
            gc.collect()

    # Deduplicate
    for day in daily_dict_conc:
        for rpm_bin in daily_dict_conc[day]:
            daily_dict_conc[day][rpm_bin] = daily_dict_conc[day][rpm_bin].drop_duplicates()

    # Save full dictionary
    if output_dir:
        key_all = 'final_daily_dict.pkl'
        if is_s3:
            bucket, prefix = output_dir[5:].split('/', 1)
            obj_key = f"{prefix}/{key_all}"
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            try:
                pd.to_pickle(daily_dict_conc, tmp_file.name)
                tmp_file.close()
                s3 = boto3.client('s3')
                s3.upload_file(tmp_file.name, bucket, obj_key)
                logger.info(f"Saved full dict to s3://{bucket}/{obj_key} | Elapsed: {time.time() - start_time:.2f}s")
            finally:
                os.unlink(tmp_file.name)
        else:
            path = os.path.join(output_dir, key_all).resolve()
            pd.to_pickle(daily_dict_conc, path)
            logger.info(f"Saved full dict to {path} | Elapsed: {time.time() - start_time:.2f}s")

    # Upload log if using S3
    if output_dir and is_s3:
        bucket, prefix = output_dir[5:].split('/', 1)
        s3 = boto3.client('s3')
        s3.upload_file(local_log, bucket, f"{prefix}/process.log")

    logger.info(f"All done! Processed {processed_files}/{total_files} files in {time.time() - start_time:.2f}s")
    return daily_dict_conc
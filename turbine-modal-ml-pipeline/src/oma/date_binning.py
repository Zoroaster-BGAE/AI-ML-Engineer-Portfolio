from datetime import datetime, timedelta
import numpy as np
import os


def extract_timestamp_from_file(filename):
    '''
    Gets a COMPANY file and returns the dt object
    '''
    timestamp_str = filename.split('_')[-1]
    return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')


def get_closest_monday(timestamp_str, previous = True, next_ = False):
    '''
    Gets in a datetime object or a string of datetime ('%Y%m%d%H%M%S') returns closest Monday
    
    If previous is true it loads the previous Monday of the week of targeted timestamp.
    
    If next_ is True AND previous False then it returns the next Monday
    
    if previous & next_ are False then the closest Monday is returned
    
    '''
    if isinstance(timestamp_str, str):
        dt = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    elif isinstance(timestamp_str, datetime):
        dt = timestamp_str
        
    
    # Calculate the previous Monday
    days_since_monday = dt.weekday()  # Monday is 0 and Sunday is 6
    previous_monday = dt - timedelta(days=days_since_monday)
    previous_monday = previous_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate the next Monday
    days_until_next_monday = (7 - days_since_monday) % 7
    if days_until_next_monday == 0:
        days_until_next_monday = 7
    next_monday = dt + timedelta(days=days_until_next_monday)
    next_monday = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Determine which Monday is closer
    if abs(dt - previous_monday) <= abs(dt - next_monday):
        closest_monday = previous_monday
    else:
        closest_monday = next_monday
    
    if previous:
        return previous_monday
    if next_:
        return next_monday
    
    return closest_monday


def get_sunday_midnight(monday_datetime):
    # Calculate the upcoming Sunday midnight
    sunday_midnight = monday_datetime + timedelta(days=6, hours=23, minutes=59, seconds=59)
 
    return sunday_midnight


def filter_files_by_week(file_name_list, target_datetime):
    '''
    Gets a file_name_list and a target_datetime (filename, str or timestamp).
    Filters the file_name_list and returns the list of the week that the target time belongs to. 
    '''
    # check if string
    if isinstance(target_datetime, str):
        # Check if filename
        if target_datetime.startswith('Structure1'):
            target_monday = extract_timestamp_from_file(target_datetime)
        else:
            target_monday = datetime.strptime(target_datetime, '%Y%m%d%H%M%S')
    
    # Check if already datetime object
    elif isinstance(target_datetime, datetime):
        target_monday = target_datetime
    
    # Get the Monday and Sunday datetimes for the target week
    #The logic works for the week target_datetime BELONGS to
    target_monday_start = get_closest_monday(target_monday, previous = True)
    target_sunday_end = get_sunday_midnight(target_monday_start)
    
    # List to store filenames belonging to the target week
    files_in_week = []
    
    for filename in file_name_list:
        # Extract timestamp from the filename        
        file_dt = extract_timestamp_from_file(filename)
        
        # Check if the file's datetime is within the target week
        if target_monday_start <= file_dt <= target_sunday_end:
            files_in_week.append(filename)
    
    return files_in_week




######_____ Not date Manipulation but binning _____######

def bin_data_by_rpm(unbined_list, statistics_dict):
    '''
    Takes unbined data and bins them by rpm
    '''
    
    rpm_dict = {}
    for file in unbined_list:
        lst = list(statistics_dict[file]['operations']['rpm'].values())
        filtered_lst = [value for value in lst if value is not None and -10 <= value < 50]
        if not filtered_lst:
            rpm_dict[file] = -1
        else:
            rpm_dict[file] = sum(filtered_lst) / len(filtered_lst)

    # Step 4: Sort dict
    sorted_dict = dict(sorted(rpm_dict.items(), key=lambda item: item[1]))


    # Step 5: Create RPM bins
    lrpm_names = ['0a', '01', '1a', '02', '2a']
    hrpm_names = ['09', '9a', '10', '10a', '11', '11a', '12', '12a', '13', '13a', '14', '14a', '15', '15a', '16', '16a']
    names = lrpm_names + hrpm_names
    bins1 = [x for x in np.arange(0.5, 3, 0.5)]
    bins = bins1 + [x for x in np.arange(9, 17.5, 0.5)]
    bins = [(x, y) for x, y in zip(bins, names)]

    # Step 6: Distribute key-value pairs into sub-dictionaries based on RPM bins
    rpm_bins_dict = {name: {} for name in names}
    for start_val, name in bins:
        rpm_bins_dict[name] = {name: rpm for name, rpm in sorted_dict.items() 
                               if start_val <= rpm < start_val + 0.5}
    
    return rpm_bins_dict


def create_weekly_rpm_bins(statistics_dict):
    '''
    Creates a dictionary with keys as the datetime of the Monday of each week in str format ('%Y%m%d')
    and values as the rpm_bins_dict of that specific week.
    '''
    all_files = list(statistics_dict.keys())
    weekly_rpm_bins = {}

    # Find the Monday of each week in the data
    unique_mondays = set(get_closest_monday(extract_timestamp_from_file(file), previous=True).strftime('%Y%m%d%H%M%S') for file in all_files)

    for monday in unique_mondays:
        # Filter files for the week starting from this Monday
        week_files = filter_files_by_week(all_files, monday)

        # Bin the RPM data for the week
        rpm_bins_dict = bin_data_by_rpm(week_files, statistics_dict)

        # Add to the weekly dictionary
        weekly_rpm_bins[monday[:-6]] = rpm_bins_dict

    return weekly_rpm_bins

def create_weekly_whole_bins(whole_bins_dict):
    '''
    Creates a dictionary with keys as the week starting date in str format ('%Y%m%d')
    and values as a nested dictionary {bin: {filename: rpm}} of that specific week.
    '''
    weekly_whole_bins = {}

    for bin_key, file_dict in whole_bins_dict.items():
        for filename, rpm in file_dict.items():
            file_dt = extract_timestamp_from_file(filename)
            week_start = get_closest_monday(file_dt, previous=True).strftime('%Y%m%d')

            if week_start not in weekly_whole_bins:
                weekly_whole_bins[week_start] = {key: {} for key in whole_bins_dict.keys()}

            weekly_whole_bins[week_start][bin_key][filename] = rpm

    return weekly_whole_bins



######_____ Load Whole Bins _____######


def list_weeks(base_folder_path):
    '''
    Loads week names of the the weekly csv collection : 
    
    usual folder : '../../data/PyOMA/whole_bin_weekly_analysis'
    
    structure : {base_folder : {XXXXXX_csv_collection (folder) : {XXXXXX_bin_conc_5DOF.csv}}}
    '''
    # List all subdirectories in the base directory
    weeks = [week for week in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, week))]
    return weeks



def load_csv_files_for_week(base_folder_path, week_folder):
    '''
    Loads a dict of structure ; {bin_num : XXXXXX_bin_conc_5DOF.csv}
    
    * The csv is turned to df in the notebook environment
    
    usual folder (base_folder_path): '../../data/PyOMA/whole_bin_weekly_analysis'
    
    structure of (base_folder_path): {base_folder : {XXXXXX_csv_collection (folder) : {XXXXXX_bin_conc_5DOF.csv}}} 
    
    week_folder : XXXXXX_csv_collection (folder)
    
    '''
    # Initialize the dictionary for the specific week
    week_dict = {}

    week_folder_path = os.path.join(base_folder_path, week_folder)
    if os.path.isdir(week_folder_path):
        # Traverse the week directory to get all CSV files
        for root, dirs, files in os.walk(week_folder_path):
            for file in files:
                if file.endswith(".csv"):
                    # Extract the week number and bin number from the file name
                    parts = file.split('_')
                    if len(parts) >= 3:  # Ensure the filename format is correct
                        week_num = parts[0]
                        bin_num = parts[1]
                        
                        # Construct the full file path
                        file_path = os.path.join(root, file)
                        
                        # Debug: Print the current file being processed
                        print(f"Processing file: {file_path}")
                        
                        # Load the CSV file into a pandas DataFrame
                        df = pd.read_csv(file_path)
                        
                        # Populate the dictionary
                        if bin_num not in week_dict:
                            week_dict[bin_num] = df
                            
    return week_dict
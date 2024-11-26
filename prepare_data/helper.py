import pandas as pd
import re
import os
import glob

def get_subdirectories(directory):
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return sorted(subdirectories)

def extract_datetime_from_tif(filename):
    """
    Extract date-time information from the filename.
    Returns: tuple includes (year, month, day, hour)
    """
    base_name = os.path.basename(filename)
    pattern1 = re.compile(r'_([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})[_A-Z]*\.tif$')
    pattern2 = re.compile(r'_([0-9]{4})([0-9]{2})([0-9]{2})\.Z([0-9]{2})([0-9]{2})[_A-Z]*\.tif$')

    match1 = pattern1.search(base_name)
    if match1:
        year, month, day, hour, *_ = match1.groups()
        return int(year), int(month), int(day), int(hour)

    match2 = pattern2.search(base_name)
    if match2:
        year, month, day, hour, _ = match2.groups()
        return int(year), int(month), int(day), int(hour)

    raise ValueError(f"Filename format not recognized: {filename}")

def list_files(directory_path, only_files=True, extension=None):
    """
    List files in a specified directory.

    Parameters:
    - directory_path (str): The path to the directory.
    - only_files (bool): If True, include only files (exclude directories).
    - extension (str): Optional extension to filter files, e.g., 'txt' for '.txt' files only.

    Returns:
    - list of str: List of file paths.
    """
    
    # Check if an extension is specified
    if extension:
        files = glob.glob(f"{directory_path}/*.{extension}")
    else:
        files = glob.glob(f"{directory_path}/*")
    
    # If only_files is True, filter out directories
    if only_files:
        files = [f for f in files if os.path.isfile(f)]
    
    return sorted(files)

#if __name__ == "__main__":
    #print(get_subdirectories("../DATA_SV/"))
    #print(get_subdirectories("../DATA_SV/Hima"))
    #print(extract_datetime_from_tif("../DATA_SV/Hima/BO4B/2019/04/01/B04B_20190401.Z0000_TB.tif"))
    #print(extract_datetime_from_tif("/home/phuc/AI/DATA_SV/Precipitation/AWS/2019/04/01/AWS_20190401000000.tif"))
    #print(list_files("/home/phuc/AI/DATA_SV/Precipitation/AWS/2019/04/01", only_files=True, extension="tif"))
    #pass

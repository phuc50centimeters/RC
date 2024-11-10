from helper import *
from pprint import pprint
import copy
import sys
import rasterio
import math
import csv

DATA_DIR = "../DATA_SV"
FEATURE = "Hima"
LABEL = "Precipitation/AWS"

# Get hima band
bands_keep_track = get_subdirectories(f"{DATA_DIR}/{FEATURE}")

days_in_bands = {}
days_in_aws = {}

# Band processing
for band in bands_keep_track:
    days_in_bands[band] = {}
    years_in_band = get_subdirectories(f"{DATA_DIR}/{FEATURE}/{band}")
    for year in years_in_band:
        days_in_bands[band][year] = {}
        months_in_band = get_subdirectories(f"{DATA_DIR}/{FEATURE}/{band}/{year}")
        for month in months_in_band:
            days_in_band = get_subdirectories(f"{DATA_DIR}/{FEATURE}/{band}/{year}/{month}")
            days_in_bands[band][year][month] = days_in_band

# DEBUG

# AWS processing
years_in_aws = get_subdirectories(f"{DATA_DIR}/{LABEL}")
for year in years_in_aws:
    days_in_aws[year] = {}
    months_in_aws = get_subdirectories(f"{DATA_DIR}/{LABEL}/{year}")
    for month in months_in_aws:
        days = get_subdirectories(f"{DATA_DIR}/{LABEL}/{year}/{month}")
        days_in_aws[year][month] = days

# Get days to keep track
# Dictionary to track year month day
days_to_work = {}

for year in days_in_aws:
    days_to_work[year] = {}
    for month in days_in_aws[year]: 
        days_to_work[year][month] = days_in_aws[year][month]
        for band in days_in_bands:
            if days_in_bands[band][year][month]:
                # Get day intersection
                intersection_day = set(days_to_work[year][month]).intersection(days_in_bands[band][year][month])
                days_to_work[year][month] = sorted(list(intersection_day))

# Dictionary to keep track hour to work
hours_to_work = {}

for year in days_to_work:
    hours_to_work[year] = {}
    for month in days_to_work[year]:
        hours_to_work[year][month] = {}
        for day in days_to_work[year][month]:
            hours_to_work[year][month][day] = sorted([extract_datetime_from_tif(file)[3] for file in list_files(f"{DATA_DIR}/{LABEL}/{year}/{month}/{day}", only_files=True, extension="tif")])
            for band in days_in_bands:
                hours_in_day_in_band = sorted([extract_datetime_from_tif(file)[3] for file in list_files(f"{DATA_DIR}/{FEATURE}/{band}/{year}/{month}/{day}", only_files=True, extension="tif")])

                intersection_hour = set(hours_to_work[year][month][day]).intersection(hours_in_day_in_band)
                hours_to_work[year][month][day] = sorted(list(intersection_hour))

#pprint(hours_to_work)

filter_hours_to_work = copy.deepcopy(hours_to_work)
# Filter data
for year in hours_to_work:
    for month in hours_to_work[year]:
        for day in hours_to_work[year][month]:
            if not hours_to_work[year][month][day]:
                del filter_hours_to_work[year][month][day]

pprint(filter_hours_to_work)

# Process tif file
for year in filter_hours_to_work:
    for month in filter_hours_to_work[year]:
        for day in filter_hours_to_work[year][month]:
            # Create new folder to store
            new_folder_path = f"./output_csv/{year}_{month}_{day}"
            os.makedirs(new_folder_path, exist_ok=True)

            # Process for himawari data
            for band in bands_keep_track:
                # tif format "B04B_20190401.Z0000_TB.tif"
                with open(f"{new_folder_path}/{band}_{year}{month}{day}.csv", mode="w", newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    # Write header
                    writer.writerow(["row", "col", "year", "month", "day", "hour", band])

                    # Loop through hour
                    for hour in filter_hours_to_work[year][month][day]:
                        with rasterio.open(f"{DATA_DIR}/{FEATURE}/{band}/{year}/{month}/{day}/{band}_{year}{month}{day}.Z{hour:02}00_TB.tif") as src:
                            array = src.read(1) # Read the first band

                        for row in range(array.shape[0]):
                            for col in range(array.shape[1]):
                                value = array[row][col]

                                # Check valid value
                                if math.isnan(value) or value == float('-inf') or value == -9999:
                                    continue
                                writer.writerow([row, col, year, month, day, hour, value])
            # Process aws data            
            with open(f"{new_folder_path}/AWS_{year}{month}{day}.csv", mode="w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                # Write header
                writer.writerow(["row", "col", "year", "month", "day", "hour", "AWS"])
                # Loop through hour
                for hour in filter_hours_to_work[year][month][day]:
                    # AWS format: AWS_20190401000000.tif
                    with rasterio.open(f"{DATA_DIR}/{LABEL}/{year}/{month}/{day}/AWS_{year}{month}{day}{hour:02}0000.tif") as src:
                        array = src.read(1)
                    for row in range(array.shape[0]):
                        for col in range(array.shape[1]):
                            value = array[row][col]

                            # Check valid value
                            if math.isnan(value) or value == float('-inf') or value == -9999:
                                continue
                            writer.writerow([row, col, year, month, day, hour, value])

print("---COMPLETE SAVE TO CSV---")

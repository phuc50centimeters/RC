import os
import pandas as pd

# File content all csv file
folder_path = "../prepare_data/train_data"

# The columns will be drop in training process
drop_columns = ['row', 'col', 'year', 'month', 'day', 'hour']

# Read and merge all csv
data_frames = []
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df = df.drop(columns=drop_columns, errors='ignore')
        data_frames.append(df)

# Concatenate all data frames in the list
combined_df = pd.concat(data_frames, ignore_index=True)

# Save the combined data frame to a CSV file
combined_df.to_csv('combined_data.csv', index=False)

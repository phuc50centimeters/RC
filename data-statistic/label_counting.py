import os
import pandas as pd

# Specify the path to your folder containing the CSV files
folder_path = '../prepare_data/train_data/'  # Replace with your actual folder path

# Loop through each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Count total lines (excluding header)
        total_lines = len(df)
        
        # Count occurrences of labels 0 and 1
        if 'Label' in df.columns:
            label_counts = {
                0: (df['Label'] == 0).sum(),
                1: (df['Label'] == 1).sum()
            }
        else:
            label_counts = {0: 0, 1: 0}

        # Print results for each file
        print(f'File: {file_name}')
        print(f'  Total lines: {total_lines}')
        print(f'  Count of Label 0: {label_counts[0]}')
        print(f'  Count of Label 1: {label_counts[1]}')
        print()  # Blank line for readability


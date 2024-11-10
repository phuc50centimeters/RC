from helper import *
import pandas as pd

csv_dirs = get_subdirectories("./output_csv")

# Auto join for each csv
for csv_dir in csv_dirs:
    csv_files = list_files(f"./output_csv/{csv_dir}", only_files=True, extension="csv")
    others_csv = []
    aws_csv = ""
    for csv_file in csv_files:
        if "AWS" in csv_file:
            aws_csv = csv_file            
        else:
            others_csv.append(csv_file)
    
    # Extract year, month, and day from the AWS file name
    filename = os.path.basename(aws_csv)
    year, month, day = filename.split('_')[1][:4], filename.split('_')[1][4:6], filename.split('_')[1][6:8]

    # Load the AWS data
    aws_df = pd.read_csv(aws_csv)

    # Peform inner join
    merged_df = aws_df
    for file in others_csv:
        other_df = pd.read_csv(file)

        merged_df = pd.merge(
            merged_df,
            other_df,
            on=['row', 'col', 'year', 'month', 'day', 'hour'],
            how='inner' # Keep only matching rows
        )
    
    # Define output file name
    output_file_name = f"joined_{year}{month}{day}.csv"
    output_file_path = os.path.join("joined_csv", output_file_name)

    # Saved merged DataFrame to the output file
    merged_df.to_csv(output_file_path, index=False)
    print(f"Final merged file saved to: {output_file_path}")

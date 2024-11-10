from helper import *

# Convert AWS col to the Label col for train

files = list_files("joined_csv_v1", only_files=True, extension='csv')
print(files)

for file in files:
    df = pd.read_csv(file)

    # Create Label col based on AWS
    df['Label'] = df['AWS'].apply(lambda x: 1 if x > 0 else 0)

    df = df.drop(columns=['AWS'])

    output_path = f"train_data_v1/{os.path.basename(file)}"
    df.to_csv(output_path, index=False)

    print("Convert sucessfully")

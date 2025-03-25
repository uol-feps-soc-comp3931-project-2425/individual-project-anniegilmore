import os
from pathlib import Path
import shutil
import pandas as pd
from glob import glob

from utils import make_path

DATASET_PATH = Path("../aptos-2019-dataset")

def combine_csv_files(path_to_csv: Path) -> None:
    # Set the folder containing the CSV files
    folder_path = f"{path_to_csv}/labels"  # Change this to your folder path

    # Get a list of all CSV files in the folder
    csv_files = glob(os.path.join(folder_path, "*.csv"))

    # Read and concatenate all CSV files
    df_list = [pd.read_csv(file) for file in csv_files]  # Read each CSV into a DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)  # Combine into one DataFrame

    # Save the merged file
    output_path = os.path.join(path_to_csv, "whole_dataset.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"Merged {len(csv_files)} CSV files into {output_path}")
    
def combine_images(path_to_images: Path) -> None:
    make_path(Path(f"{DATASET_PATH}/images"))
    for folder in ["resized test 15", "resized train 15"]:
        images = [f for f in os.listdir(f"{DATASET_PATH}/{folder}") if f.lower().endswith(".jpg")]
        for image in images:
            shutil.copy(Path(f"{DATASET_PATH}/{folder}") / image, Path(f"{DATASET_PATH}/images/{image}"))
    
if __name__ == "__main__":
    # combine_csv_files(DATASET_PATH)
    combine_images(DATASET_PATH)
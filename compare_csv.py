from pathlib import Path
from constants import DATASET_PATH

TRAINING_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/trainLabels.csv")
VALIDATION_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/validateLabels.csv")

with (
    open(TRAINING_ANNOTATIONS_PATH, "r") as t1,
    open(VALIDATION_ANNOTATIONS_PATH, "r") as t2,
):
    fileone = t1.readlines()
    filetwo = t2.readlines()

with open("similarity.csv", "w") as outFile:
    for line in filetwo:
        if line in fileone:
            outFile.write(line)

with open("difference.csv", "w") as out2File:
    for line in filetwo:
        if line not in fileone:
            out2File.write(line)

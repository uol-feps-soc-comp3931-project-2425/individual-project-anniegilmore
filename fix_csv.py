import csv
import os

inputFileName = "trainLabels.csv"
outputFileName = os.path.splitext(inputFileName)[0] + "_modified.csv"

files = os.listdir("../messidor-1/dataset_0.7")
for file in files:
    if file.endswith("trainLabels.csv"):
        with (
            open(f"../messidor-1/dataset_0.7/{file}", "r") as inFile,
            open(
                f"../messidor-1/dataset_0.7/{file.removesuffix('.csv')}_mod.csv",
                "w",
                newline="",
            ) as outfile,
        ):
            r = csv.DictReader(inFile)
            dataset_reader = csv.reader(inFile, delimiter=",")
            w = csv.DictWriter(outfile, ["image", "level"])
            # write new header
            w.writeheader()

            # copy the rest
            for index, row in enumerate(dataset_reader):
                if index > 0:
                    w.writerow({"image": row[0], "level": row[2]})

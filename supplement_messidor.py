import csv
from pathlib import Path
import random
import cv2

APTOS_PATH = Path("../aptos-2019-dataset")
MESSIDOR_PATH = Path("../messidor-1")
SUPP_LABELS = Path(f"{MESSIDOR_PATH}/supplemented/supplementedLabels.csv")

def get_aptos_level_distribution_map(path_to_dataset: Path) -> dict[str, list[str]]:
    level_image_map: dict[str, list[str]] = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4": [],
    }
    with open(path_to_dataset, "r", newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        for rows in dataset_reader:
            for level in level_image_map.keys():
                if rows[1] == level:
                    level_image_map[level].append(rows[0])
    return level_image_map

def get_messidor_level_distribution_map(path_to_dataset: Path) -> dict[str, list[str]]:
    level_image_map: dict[str, list[str]] = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
    }
    with open(path_to_dataset, "r", newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        for rows in dataset_reader:
            for level in level_image_map.keys():
                if rows[2] == level:
                    level_image_map[level].append(rows[0])
    return level_image_map

aptos_dsitribution = get_aptos_level_distribution_map(f"{APTOS_PATH}/whole_dataset.csv")
messidor_distribution = get_messidor_level_distribution_map(f"{MESSIDOR_PATH}/annotations.csv")

num_images_per_level = 2250

with open(SUPP_LABELS, "a", newline="") as file:
    writer = csv.DictWriter(file, ["image","Ophthalmologic department","level","me risk"])
    writer.writeheader()

for level in messidor_distribution.keys():
    num_messidor_images = len(messidor_distribution[level])
    num_aptos_needed = num_images_per_level - num_messidor_images
    num_aptos_fulfilled = 0
    while num_aptos_fulfilled < num_aptos_needed:
        aptos_image = random.choice(aptos_dsitribution[level])
        img = cv2.imread(f"{APTOS_PATH}/images/{aptos_image}.jpg")
        cv2.imwrite(f"{MESSIDOR_PATH}/supplemented/images/{aptos_image}.jpg", img)
        with open(SUPP_LABELS, "a", newline="") as file:
            writer = csv.DictWriter(file, ["image","Ophthalmologic department","level","me risk"])
            writer.writerow({"image": f"{aptos_image}.jpg", "Ophthalmologic department": "N/A", "level": level, "me risk": "N/A"})
    print(level, num_aptos_needed)
    print(f"There are {len(aptos_dsitribution[level])} available aptos images for level {level}")
    

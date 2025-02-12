import csv
from pathlib import Path
import random

BASE_PATH = Path("dataset")
DATASET_PATH = Path(f"{BASE_PATH}/trainLabels.csv")
CROPPED_DATASET_PATH = Path(f"{BASE_PATH}/trainLabels_cropped.csv")


def get_dataset_reader(path_to_dataset: Path):
    with open(path_to_dataset, newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
    return dataset_reader


def get_level_distribution_map(path_to_dataset: Path) -> dict[str, list[str]]:
    level_image_map: dict[str, list[str]] = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4": [],
    }
    with open(path_to_dataset, newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        for rows in dataset_reader:
            for level in level_image_map.keys():
                if rows[1] == level:
                    level_image_map[level].append(rows[0])
    return level_image_map


def standardize_dataset(path_to_dataset: Path) -> None:
    level_image_map: dict[str, list[str]] = get_level_distribution_map(path_to_dataset)
    min_count: int = len(level_image_map["0"])
    for level_images in level_image_map.values():
        no_images = len(level_images)
        if no_images < min_count:
            min_count = no_images
    standardised_level_image_map: dict[str, list[str]] = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4": [],
    }
    dict_to_write: list[dict[str, str]] = []
    for level_to_fill in standardised_level_image_map:
        count: int = 0
        while count != min_count:
            dict_to_write.append(
                {
                    "image": random.choice(level_image_map[level_to_fill]),
                    "level": level_to_fill,
                }
            )
            count += 1
    with open(f"{BASE_PATH}/standardised_trainLabels.csv", "w", newline="") as csvfile:
        dict_writer = csv.DictWriter(csvfile, fieldnames=["image", "level"])
        dict_writer.writeheader()
        dict_writer.writerows(dict_to_write)


if __name__ == "__main__":
    count = get_level_distribution_map(DATASET_PATH)
    print(get_level_distribution_map(DATASET_PATH))
    standardize_dataset(DATASET_PATH)

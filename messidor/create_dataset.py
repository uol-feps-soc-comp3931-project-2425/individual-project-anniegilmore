import csv
import os
from pathlib import Path
import random
import shutil
import pandas as pd

BASE_PATH = Path("../messidor-1")
DATASET_PATH = Path(f"{BASE_PATH}/trainLabels.csv")
OG_ANNOTATIONS_PATH = Path(f"{BASE_PATH}/annotations.csv")
USING_ORIGINAL_DISTRIBUTION = False
BASE_LIST = [
    "Base11",
    "Base12",
    "Base13",
    "Base14",
    "Base21",
    "Base22",
    "Base23",
    "Base24",
    "Base31",
    "Base32",
    "Base33",
    "Base34",
]


def make_path(path_to_create: Path) -> None:
    Path.mkdir(path_to_create, parents=True, exist_ok=True)


def combine_annotations(path_to_new_annotations: Path) -> None:
    excel_list: list = []
    for base in BASE_LIST:
        df = pd.read_excel(f"{BASE_PATH}/Annotation_{base}.xls", header=0)
        excel_list.append(df)
    merged = pd.concat(excel_list, ignore_index=True)

    merged.to_csv(path_to_new_annotations, index=False)


def combine_images(path_to_new_images: Path) -> None:
    make_path(path_to_new_images)
    for base in BASE_LIST:
        for image in os.listdir(f"{BASE_PATH}/{base}"):
            shutil.copy(Path(f"{BASE_PATH}/{base}") / image, path_to_new_images)


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
    }
    with open(path_to_dataset, "r", newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        for rows in dataset_reader:
            for level in level_image_map.keys():
                if rows[1] == level:
                    level_image_map[level].append(rows[0])
    return level_image_map


def split_data(
    original_annotations: Path, data_to_split: list[str], percentage_train: float
) -> None:
    new_dataset_path: str = f"{BASE_PATH}/dataset_{percentage_train}"
    make_path(Path(new_dataset_path))
    make_path(Path(f"{new_dataset_path}/images"))
    original_data_dir: Path = Path(f"{BASE_PATH}/images")
    num_validate_images: int = int(len(data_to_split) * (1 - percentage_train))
    num_train_images: int = int(len(data_to_split) - num_validate_images)
    validation_images = move_images(
        num_validate_images,
        data_to_split,
        original_data_dir,
        Path(f"{new_dataset_path}/images"),
    )
    train_images = move_images(
        num_train_images,
        data_to_split,
        original_data_dir,
        Path(f"{new_dataset_path}/images"),
    )
    write_moved_annotations(
        original_annotations,
        validation_images,
        Path(f"{new_dataset_path}/validateLabels.csv"),
    )
    write_moved_annotations(
        original_annotations, train_images, Path(f"{new_dataset_path}/trainLabels.csv")
    )


def move_images(
    num_images_to_move: int, image_names: list[str], original_dir: Path, new_dir: Path
) -> list[str]:
    moved_images: list[str] = []
    for _ in range(num_images_to_move):
        image_to_be_moved: str = random.choice(image_names)
        # shutil.copy(original_dir / image_to_be_moved, new_dir)
        image_names.remove(image_to_be_moved)
        moved_images.append(image_to_be_moved)
    return moved_images


def write_csv_rows(annotations_file: Path, annotations: dict) -> None:
    with open(annotations_file, "a", newline="") as file:
        writer = csv.DictWriter(file, annotations.keys())
        writer.writerow(annotations)


def write_csv_header(annotations_file: Path) -> None:
    with open(annotations_file, "w", newline="") as file:
        writer = csv.DictWriter(
            file, ["image", "Ophthalmologic department", "level", "me risk"]
        )
        writer.writeheader()


def write_moved_annotations(
    annotations_path: Path, moved_images: list[str], annotations_file: Path
) -> None:
    write_csv_header(annotations_file)
    with open(annotations_path, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in annotations:
            if row["Image name"] in moved_images:
                write_csv_rows(annotations_file, row)


def reduce_dataset(
    image_map: dict[str, list[str]], max_class: int
) -> dict[str, list[str]]:
    new_image_map: dict[str, list[str]] = {"0": [], "1": [], "2": [], "3": [], "4": []}
    for level in image_map:
        count: int = 0
        while count != max_class and len(image_map[level]) > 0:
            image = random.choice(image_map[level])
            image_map[level].remove(image)
            new_image_map[level].append(image)
            count += 1
    return new_image_map


if __name__ == "__main__":
    # combine_annotations(f"{BASE_PATH}/annotations.csv")
    # combine_images(Path(f"{BASE_PATH}/images"))

    with open(f"{BASE_PATH}/annotations.csv", "r", newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        data: list[str] = [f"{row[0]}" for row in dataset_reader]
        del data[0]

    data = os.listdir("../messidor-1/dataset_0.8/processed_images")
    # print(data)
    split_data(Path(f"{BASE_PATH}/annotations.csv"), data, 0.8)

import csv
from pathlib import Path
import random
import shutil
from utils import make_path

BASE_PATH = Path("../individual-project-dataset")
DATASET_PATH = Path(f"{BASE_PATH}/trainLabels.csv")
CROPPED_DATASET_PATH = Path(f"{BASE_PATH}/trainLabels_cropped.csv")
OG_DATASET_PATH = Path(f"{BASE_PATH}/untouched_dataset")
OG_ANNOTATIONS_PATH = Path(f"{OG_DATASET_PATH}/trainLabels.csv")
USING_ORIGINAL_DISTRIBUTION = False


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
    with open(path_to_dataset, "r", newline="") as csvfile:
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
    with open("dataset/standardised_trainLabels.csv", "w", newline="") as csvfile:
        dict_writer = csv.DictWriter(csvfile, fieldnames=["image", "level"])
        dict_writer.writeheader()
        dict_writer.writerows(dict_to_write)


def split_data(data_to_split: list[str], percentage_train: float) -> None:
    new_dataset_path: str = f"{BASE_PATH}/dataset_{percentage_train}"
    make_path(Path(new_dataset_path))
    make_path(Path(f"{new_dataset_path}/images"))
    original_data_dir: Path = Path(f"{OG_DATASET_PATH}/resized_train/resized_train")
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
        validation_images, Path(f"{new_dataset_path}/validateLabels.csv")
    )
    write_moved_annotations(train_images, Path(f"{new_dataset_path}/trainLabels.csv"))


def move_images(
    num_images_to_move: int, image_names: list[str], original_dir: Path, new_dir: Path
) -> list[str]:
    moved_images: list[str] = []
    for _ in range(num_images_to_move):
        image_to_be_moved: str = random.choice(image_names)
        shutil.copy(original_dir / image_to_be_moved, new_dir)
        image_names.remove(image_to_be_moved)
        moved_images.append(image_to_be_moved.removesuffix(".jpeg"))
    return moved_images


def write_csv_rows(annotations_file: Path, annotations: dict) -> None:
    with open(annotations_file, "a", newline="") as file:
        writer = csv.DictWriter(file, annotations.keys())
        writer.writerow(annotations)


def write_csv_header(annotations_file: Path) -> None:
    with open(annotations_file, "w", newline="") as file:
        writer = csv.DictWriter(file, ["image", "level"])
        writer.writeheader()


def write_moved_annotations(moved_images: list[str], annotations_file: Path) -> None:
    write_csv_header(annotations_file)
    with open(OG_ANNOTATIONS_PATH, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in annotations:
            if row["image"] in moved_images:
                write_csv_rows(annotations_file, row)
                

if __name__ == "__main__":
    count = get_level_distribution_map(OG_ANNOTATIONS_PATH)
    standardize_dataset(OG_ANNOTATIONS_PATH)
    if USING_ORIGINAL_DISTRIBUTION:
        data: list[str] = [
            f"{image_name.stem}.jpeg" for image_name in Path.glob(Path(f"{OG_DATASET_PATH}/resized_train/resized_train"), "*.jpeg")
        ]
    else:
        with open("dataset/standardised_trainLabels.csv", "r", newline="") as csvfile:
            dataset_reader = csv.reader(csvfile, delimiter=",")
            data: list[str] = [f"{row[0]}.jpeg" for row in dataset_reader]
            del(data[0])
    split_data(data, 0.8)

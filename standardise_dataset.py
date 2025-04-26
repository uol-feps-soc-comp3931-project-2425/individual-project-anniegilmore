import csv
import os
from pathlib import Path
import random
import shutil
from utils import make_path

# BASE_PATH = Path("../individual-project-dataset")
BASE_PATH = Path("../messidor-1")
DATASET_PATH = Path(f"{BASE_PATH}/trainLabels.csv")
CROPPED_DATASET_PATH = Path(f"{BASE_PATH}/trainLabels_cropped.csv")
OG_DATASET_PATH = Path(f"{BASE_PATH}/untouched_dataset")
# OG_ANNOTATIONS_PATH = Path(f"{OG_DATASET_PATH}/trainLabels.csv")
OG_ANNOTATIONS_PATH = Path(f"{BASE_PATH}/annotations.csv")
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
    }
    with open(path_to_dataset, "r", newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        for rows in dataset_reader:
            for level in level_image_map.keys():
                if rows[1] == level:
                    level_image_map[level].append(rows[0])
    return level_image_map


def standardize_dataset(path_to_dataset: Path) -> str:
    path_to_labels: str = "dataset/standardised_trainLabels.csv"
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
    chosen_images: list[str] = []
    for level_to_fill in standardised_level_image_map:
        count: int = 0
        while count != min_count:
            image = random.choice(level_image_map[level_to_fill])
            if image not in chosen_images:
                chosen_images.append(image)
                dict_to_write.append(
                    {
                        "image": image,
                        "level": level_to_fill,
                    }
                )
                count += 1
            else:
                continue
    with open(path_to_labels, "w", newline="") as csvfile:
        dict_writer = csv.DictWriter(csvfile, fieldnames=["image", "level"])
        dict_writer.writeheader()
        dict_writer.writerows(dict_to_write)
    return path_to_labels


def split_data(original_annotations: Path, data_to_split: list[str]) -> None:
    new_dataset_path: str = f"{BASE_PATH}/dataset_0.7"
    make_path(Path(new_dataset_path))
    make_path(Path(f"{new_dataset_path}/images"))
    original_data_dir: Path = Path(f"{BASE_PATH}/images")
    num_train_images: int = int(len(data_to_split) * 0.7)
    num_validate_images: int = int(len(data_to_split) * 0.2)
    num_test_images: int = len(data_to_split) - num_train_images - num_validate_images
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
    test_images = move_images(
        num_test_images,
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
    write_moved_annotations(
        original_annotations, test_images, Path(f"{new_dataset_path}/testLabels.csv")
    )


def move_images(
    num_images_to_move: int, image_names: list[str], original_dir: Path, new_dir: Path
) -> list[str]:
    moved_images: list[str] = []
    for _ in range(num_images_to_move):
        image_to_be_moved: str = random.choice(image_names)
        shutil.copy(original_dir / image_to_be_moved, new_dir)
        image_names.remove(image_to_be_moved)
        moved_images.append(image_to_be_moved)
    return moved_images


def write_csv_rows(annotations_file: Path, annotations: dict) -> None:
    with open(annotations_file, "a", newline="") as file:
        writer = csv.DictWriter(file, annotations.keys())
        writer.writerow(annotations)


def write_csv_header(annotations_file: Path) -> None:
    with open(annotations_file, "w", newline="") as file:
        writer = csv.DictWriter(file, ["image", "level"])
        writer.writeheader()


def write_moved_annotations(
    annotations_path: Path, moved_images: list[str], annotations_file: Path
) -> None:
    write_csv_header(annotations_file)
    with open(annotations_path, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in annotations:
            if f"{row['Image name']}" in moved_images:
                write_csv_rows(annotations_file, row)


def group_images_into_3_classes(path_to_dataset: Path) -> str:
    # path_to_labels: str = "dataset/3_class_classification.csv"
    path_to_labels: str = f"{BASE_PATH}/3_class_classification.csv"
    level_image_map = get_level_distribution_map(path_to_dataset)
    grouped_image_map: dict[str, list[str]] = {
        "0": [],
        "1": [],
        "2": [],
    }
    dict_to_write: list[dict[str, str]] = []
    grouped_image_map["0"] = level_image_map["0"]
    grouped_image_map["1"] = level_image_map["1"] + level_image_map["2"]
    grouped_image_map["2"] = level_image_map["3"] + level_image_map["4"]
    reduced_image_map = reduce_dataset(grouped_image_map, 300)
    for level in reduced_image_map:
        for image in reduced_image_map[level]:
            dict_to_write.append(
                {
                    "image": image,
                    "level": level,
                }
            )
    random.shuffle(dict_to_write)
    with open(path_to_labels, "w", newline="") as csvfile:
        dict_writer = csv.DictWriter(csvfile, fieldnames=["image", "level"])
        dict_writer.writeheader()
        dict_writer.writerows(dict_to_write)
    print(f"In class 0: {len(reduced_image_map['0'])}")
    print(f"In class 1: {len(reduced_image_map['1'])}")
    print(f"In class 2: {len(reduced_image_map['2'])}")
    return path_to_labels


def reduce_dataset(
    image_map: dict[str, list[str]], max_class: int
) -> dict[str, list[str]]:
    new_image_map: dict[str, list[str]] = {"0": [], "1": [], "2": []}
    for level in image_map:
        count: int = 0
        while count != max_class and len(image_map[level]) > 0:
            image = random.choice(image_map[level])
            image_map[level].remove(image)
            new_image_map[level].append(image)
            count += 1
    return new_image_map


def choose_validation_images(percentage_train: float) -> None:
    # level_dist_map = get_level_distribution_map(OG_ANNOTATIONS_PATH)
    print(os.listdir("../aptos-2019-dataset"))
    level_dist_map = get_level_distribution_map(
        "../aptos-2019-dataset/whole_dataset.csv"
    )
    # with open(OG_ANNOTATIONS_PATH, "r", newline="") as file:
    #     annotations: csv.DictReader[str] = csv.DictReader(file)
    #     for row in annotations:
    #         if f"{row['image']}.jpg" in moved_images:
    #             write_csv_rows(annotations_file, row)
    # print(level_dist_map)

    total_images = 0
    for level in level_dist_map:
        total_images += len(level_dist_map[level])
    num_val_images = (1 - percentage_train) * total_images
    print(f"We need {num_val_images} val images")

    print(f"There are {len(level_dist_map['0'])} images with label 0")
    print(f"There are {len(level_dist_map['1'])} images with label 1")
    print(f"There are {len(level_dist_map['2'])} images with label 2")
    print(f"There are {len(level_dist_map['3'])} images with label 3")
    print(f"There are {len(level_dist_map['4'])} images with label 4")

    num_images_one_class = num_val_images / 4
    print(num_images_one_class)


if __name__ == "__main__":
    # count = get_level_distribution_map(OG_ANNOTATIONS_PATH)
    # path_to_labels = standardize_dataset(OG_ANNOTATIONS_PATH)

    path_to_labels = "../messidor-1/annotations.csv"

    if USING_ORIGINAL_DISTRIBUTION:
        data: list[str] = [
            f"{image_name.stem}.jpg"
            for image_name in Path.glob(
                Path(f"{OG_DATASET_PATH}/resized_train/resized_train"), "*.jpeg"
            )
        ]
    else:
        with open(path_to_labels, "r", newline="") as csvfile:
            dataset_reader = csv.reader(csvfile, delimiter=",")
            data: list[str] = [f"{row[0]}" for row in dataset_reader]
            del data[0]

    split_data(Path(path_to_labels), data)

import csv
from pathlib import Path
import random
from utils import make_path

BASE_PATH = Path("../messidor-1")


def write_csv_header(annotations_file: Path) -> None:
    """Writes header of new subset csv file"""
    with open(annotations_file, "w", newline="") as file:
        writer = csv.DictWriter(file, ["image", "level"])
        writer.writeheader()
    return


def write_csv_rows(annotations_file: Path, annotations: dict) -> None:
    """Writes annotations of new subset csv file"""
    with open(annotations_file, "a", newline="") as file:
        writer = csv.DictWriter(file, annotations.keys())
        writer.writerow(annotations)
    return


def write_moved_annotations(
    annotations_path: Path, images: list[str], new_annotations_path: Path
) -> None:
    """Creates subsets by writing new annotations file for each subset"""
    write_csv_header(new_annotations_path)
    with open(annotations_path, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in annotations:
            if f"{row['Image name']}" in images:
                write_csv_rows(new_annotations_path, row)
    return


def choose_images(
    num_images_to_choose: int, image_names: list[str]
) -> tuple[list[str], list[str]]:
    """Randomly chooses images to make up subset"""
    chosen_images: list[str] = []
    for _ in range(num_images_to_choose):
        chosen_image: str = random.choice(image_names)
        image_names.remove(chosen_image)
        chosen_images.append(chosen_image)
    return image_names, chosen_images


def split_data(
    original_annotations: Path,
    data_to_split: list[str],
    train_percent: float = 0.7,
    val_percent: float = 0.2,
) -> None:
    """Calculates number of images in each subset
    Randomly chooses images to make up each subset
    Writes new annotations file for each subset"""
    new_dataset_path: str = f"{BASE_PATH}/dataset_{train_percent}"
    make_path(Path(new_dataset_path))
    make_path(Path(f"{new_dataset_path}/images"))
    num_train_images: int = int(len(data_to_split) * train_percent)
    num_validate_images: int = int(len(data_to_split) * val_percent)
    num_test_images: int = len(data_to_split) - num_train_images - num_validate_images
    remaining_images, train_images = choose_images(
        num_train_images,
        data_to_split,
    )
    remaining_images, validation_images = choose_images(
        num_validate_images,
        remaining_images,
    )
    assert len(remaining_images) == num_test_images
    test_images: list[str] = remaining_images
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
    return


def split_dataset() -> None:
    """Loads in image names in dataset and splits dataset into train, validation and test"""
    path_to_labels: Path = Path("../messidor-1/annotations.csv")
    with open(path_to_labels, "r", newline="") as csvfile:
        dataset_reader = csv.reader(csvfile, delimiter=",")
        data: list[str] = [f"{row[0]}" for row in dataset_reader]
        del data[0]
    split_data(Path(path_to_labels), data)


if __name__ == "__main__":
    split_dataset()

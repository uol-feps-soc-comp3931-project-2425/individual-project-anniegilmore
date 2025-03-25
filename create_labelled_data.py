
import csv
import os
from pathlib import Path
import random
import shutil

from utils import make_path

DATA_PATH = Path("../Cifar10/cifar10")

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
        
def format_dataset() -> None:
    make_path(Path(f"{DATA_PATH}/images"))
    for folder in ["test", "train"]:
        level2_folders = os.listdir(os.path.join(DATA_PATH, folder))
        label_data: list[dict[str, str]] = []
        for lvl2_folder in level2_folders:
            images = [f for f in os.listdir(f"{DATA_PATH}/{folder}/{lvl2_folder}") if f.lower().endswith(".png")]
            for image in images:
                label_data.append({"image": f"{folder}_{lvl2_folder}_{image}", "level": lvl2_folder})
                shutil.copy(Path(f"{DATA_PATH}/{folder}/{lvl2_folder}") / image, Path(f"{DATA_PATH}/images/{folder}_{lvl2_folder}_{image}"))
                
        with open(Path(f"../Cifar10/cifar10/{folder}Labels.csv"), "w", newline="") as csvfile:
            dict_writer = csv.DictWriter(csvfile, fieldnames=["image", "level"])
            dict_writer.writeheader()
            dict_writer.writerows(label_data)
                

                
if __name__ == "__main__":
    format_dataset()
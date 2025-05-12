import csv
from pathlib import Path
import random

import numpy as np
import cv2
from cv2.typing import MatLike

MAX_ROTATION_ANGLE = 15
LEVELS_TO_AUGMENT = [0, 1, 2, 3]

DATASET_PATH = Path("dataset")
TRAINING_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/trainLabels.csv")

GEOMETRIC_AUGS = ["hflip", "vflip", "rotate", "zoom"]
PHOTOMETRIC_AUGS = ["saturation", "contrast", "brightness"]


def horizontal_flip(image: MatLike) -> MatLike:
    """function to apply horizontal flip augmentation"""
    return cv2.flip(image, 1)


def vertical_flip(image: MatLike) -> MatLike:
    """function to apply vertical flip augmentation"""
    return cv2.flip(image, 0)


def rotate(image: MatLike) -> MatLike:
    """function to apply rotation augmentation of random angle"""
    chosen_angle: int = int(random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE))
    height, width = image.shape[:2]
    rotation_matrix: MatLike = cv2.getRotationMatrix2D(
        (int(width / 2), int(height / 2)), chosen_angle, 1
    )
    rotated_image: MatLike = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def zoom(image: MatLike) -> MatLike:
    """function to apply zoom augmentation of random zoom factor"""
    zoom: int = 1
    while zoom == 1:
        zoom = random.uniform(0.9, 1.1)
    centre_y, centre_x = [dimension / 2 for dimension in image.shape[:-1]]
    rotation_matrix = cv2.getRotationMatrix2D((centre_x, centre_y), 0, zoom)
    zoomed_image = cv2.warpAffine(
        image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )
    return zoomed_image


def adjust_saturation(image: MatLike) -> MatLike:
    """function to apply saturation adjustment augmentation of random strength"""
    new_saturation: int = 1
    while new_saturation == 1:
        new_saturation = random.uniform(0.5, 1.5)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    adjusted_saturation = np.clip(s_channel * new_saturation, 0, 255).astype(np.uint8)
    hsv_new: MatLike = cv2.merge([h_channel, adjusted_saturation, v_channel])
    saturation_adjusted_image: MatLike = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    return saturation_adjusted_image


def adjust_contrast(image: MatLike) -> MatLike:
    """function to apply contrast adjustment augmentation of random strength"""
    new_contrast: int = 1
    while new_contrast == 1:
        new_contrast = random.uniform(0.8, 1.2)
    contrast_adjusted_image = cv2.convertScaleAbs(image, alpha=new_contrast, beta=0)
    return contrast_adjusted_image


def adjust_brightness(image: MatLike) -> MatLike:
    """function to apply brightness adjustment augmentation of random strength"""
    new_brightness: int = 1
    while new_brightness == 1:
        new_brightness = random.uniform(0.5, 1.5)
    hsv_image: MatLike = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    adjusted_brightness = np.clip(v_channel * new_brightness, 0, 255).astype(np.uint8)
    hsv_new: MatLike = cv2.merge([h_channel, s_channel, adjusted_brightness])
    brightness_adjusted_image: MatLike = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    return brightness_adjusted_image


def get_level_distribution_map(path_to_dataset: Path) -> dict[str, list[str]]:
    """function to load current dataset class sizes and information"""
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


def apply_augmentations(augs_to_apply: list[str], image: MatLike) -> MatLike:
    """function to apply randomly chosen augmentation techniques"""
    for augmentation in augs_to_apply:
        if augmentation == "hflip":
            augmented_image = horizontal_flip(image)
        elif augmentation == "vflip":
            augmented_image = vertical_flip(image)
        elif augmentation == "rotate":
            augmented_image = rotate(image)
        elif augmentation == "zoom":
            augmented_image = zoom(image)
        elif augmentation == "brightness":
            augmented_image = adjust_brightness(image)
        elif augmentation == "saturation":
            augmented_image = adjust_saturation(image)
        elif augmentation == "contrast":
            augmented_image = adjust_contrast(image)
    return augmented_image


def choose_random_augs() -> list[str]:
    """function to choose which augmentation techniques will be applied"""
    aug_choice: str = random.choice(["photo", "geo", "both"])
    augs_to_apply: list = []
    if aug_choice == "photo":
        augs_to_apply.append(random.choice(PHOTOMETRIC_AUGS))
    elif aug_choice == "geo":
        augs_to_apply.append(random.choice(GEOMETRIC_AUGS))
    else:
        augs_to_apply.append(random.choice(PHOTOMETRIC_AUGS))
        augs_to_apply.append(random.choice(GEOMETRIC_AUGS))
    return augs_to_apply


def write_augmented_image(
    augmented_image: MatLike, applied_augs: list[str], og_image_name: str, dr_level: int
) -> None:
    """function to add augmented image to existing dataset"""
    distorted_image_name = (
        f"{og_image_name.removesuffix('.tif')}_{'_'.join(applied_augs)}.tif"
    )
    cv2.imwrite(
        f"{DATASET_PATH}/supplemented/{distorted_image_name}",
        augmented_image,
    )
    with open(TRAINING_ANNOTATIONS_PATH, "a", newline="") as file:
        writer = csv.DictWriter(file, ["image", "level"])
        row = {"image": distorted_image_name, "level": dr_level}
        writer.writerow(row)
    return


def apply_supplementary_augmentations(
    distribution_map: dict[str, list], level: int
) -> None:
    """function to augment individual class with decided augmentation techniques"""
    num_augs_to_apply: int = 2500 - len(distribution_map[level])
    num_augs_applied: int = 0
    while num_augs_applied < num_augs_to_apply:
        image_name: str = random.choice(distribution_map[level])
        if image_name.endswith("PP.tif"):
            image: MatLike = cv2.imread(f"{DATASET_PATH}/images/{image_name}")
            if image is None:
                continue
        else:
            continue
        augs_to_apply: list[str] = choose_random_augs()
        augmented_image: MatLike = apply_augmentations(augs_to_apply, image)
        write_augmented_image(augmented_image, augs_to_apply, image_name, level)
        num_augs_applied += 1
    new_distribution_map = get_level_distribution_map(TRAINING_ANNOTATIONS_PATH)
    print(
        f"There are {len(new_distribution_map[str(level)])} images of level {str(level)}"
    )


def augment_dataset(
    data_to_augment: Path = TRAINING_ANNOTATIONS_PATH,
    classes_to_augment: list[str] = LEVELS_TO_AUGMENT,
) -> None:
    """function to augment given dataset"""
    distribution_map: dict[str, list] = get_level_distribution_map(data_to_augment)
    for dr_level in classes_to_augment:
        print(
            f"There are {len(distribution_map[str(dr_level)])} images of level {str(dr_level)}"
        )
        apply_supplementary_augmentations(distribution_map, dr_level)


if __name__ == "__main__":
    augment_dataset()

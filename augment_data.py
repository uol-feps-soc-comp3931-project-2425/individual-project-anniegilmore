import csv
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm
import cv2
from standardise_dataset import get_level_distribution_map

import warnings
warnings.filterwarnings("ignore")

# DATASET_PATH = Path("../messidor-1/dataset_0.8")
# DATASET_PATH = Path("../messidor-1/dataset_0.8/test_dataset")
DATASET_PATH = Path("../messidor-1/dataset_0.7")
TRAINING_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/trainLabels.csv")

GEOMETRIC_AUGS = ["hflip", "vflip", "rotate", "zoom"]
PHOTOMETRIC_AUGS = ["saturation", "contrast", "brightness"]
POTENTIAL_AUGMENTATIONS = ["hflip", "vflip", "noise", "rotate", "zoom", "saturation", "contrast", "brightness"]

def zoom(img, angle=0, coord=None):
    zoom: int = 1
    while zoom == 1:
        zoom = random.uniform(0.9, 1.1)
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def adjust_contrast(image):
    new_contrast: int = 1
    while new_contrast == 1:
        new_contrast = random.uniform(0.8, 1.2)
    new_image = cv2.convertScaleAbs(image, alpha=new_contrast, beta=0)
    return new_image
    
def adjust_brightness(image):
    new_brightness: int = 1
    while new_brightness == 1:
        new_brightness = random.uniform(0.5, 1.5)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    vnew = np.clip(v * new_brightness, 0, 255).astype(np.uint8)
    hsv_new = cv2.merge([h, s, vnew])
    new_image = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    return new_image

def adjust_saturation(image):
    new_sat: int = 1
    while new_sat == 1:
        new_sat = random.uniform(0.5, 1.5)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    snew = np.clip(s * new_sat, 0, 255).astype(np.uint8)
    hsv_new = cv2.merge([h, snew, v])
    bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    return bgr_new


def choose_augmentations(no_augs: int) -> None:
    augs_to_apply = []
    while len(augs_to_apply) < no_augs:
        aug_to_apply = random.choice(POTENTIAL_AUGMENTATIONS)
        if aug_to_apply not in augs_to_apply:
            augs_to_apply.append(aug_to_apply)
    return augs_to_apply


def apply_augmentations(augs_to_apply: list[str], image):
    for augmentation in augs_to_apply:
        if augmentation == "hflip":
            image = horizontal_flip(image)
        elif augmentation == "vflip":
            image = vertical_flip(image)
        elif augmentation == "noise":
            image = sp_noise(image)
        elif augmentation == "rotate":
            image = rotation(image, 15)
        elif augmentation == "zoom":
            image = zoom(image)
        elif augmentation == "brightness":
            image = adjust_brightness(image)
        elif augmentation == "saturation":
            image = adjust_saturation(image)
        elif augmentation == "contrast":
            image = adjust_contrast(image)
    return image


def augment_image(num_augs, row, img) -> None:
    augs_to_apply = choose_augmentations(num_augs)
    print(augs_to_apply)
    distorted_image = apply_augmentations(augs_to_apply, img)
    distortion_string = "_".join(augs_to_apply)
    cv2.imwrite(
        f"{DATASET_PATH}/augmented_images/{row['image'].removesuffix('.tif')}_{distortion_string}.tif",
        distorted_image,
    )
    with open(TRAINING_ANNOTATIONS_PATH, "a", newline="") as file:
        writer = csv.DictWriter(file, row.keys())
        row["image"] = f"{row['image'].removesuffix('.tif')}_{distortion_string}.tif"
        writer.writerow(row)


def apply_random_augmentation_combinations() -> None:
    with open(TRAINING_ANNOTATIONS_PATH, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in tqdm(annotations):
            if row["image"].endswith("PP.tif"):
                img = cv2.imread(f"{DATASET_PATH}/processed_images/{row['image']}")
            else:
                img = None
            if img is not None:
                with open(Path(f"{DATASET_PATH}/newTrainLabels.csv"), "a", newline="") as file:
                    writer = csv.DictWriter(file, row.keys())
                    writer.writeheader()
                    writer.writerow(row)
                for i in range(1, 5):
                    augment_image(i, row, img)
                    
def get_augs() -> list[str]:
    aug_choice = random.choice(["photo", "geo", "both"])
    augs_to_apply = []
    if aug_choice == "photo":
        augs_to_apply.append(random.choice(PHOTOMETRIC_AUGS))
    elif aug_choice == "geo":
        augs_to_apply.append(random.choice(GEOMETRIC_AUGS))
    else:
        augs_to_apply.append(random.choice(PHOTOMETRIC_AUGS))
        augs_to_apply.append(random.choice(GEOMETRIC_AUGS))
    return augs_to_apply
    
                    
def supplementary_augmentations() -> None:
    distribution_map = get_level_distribution_map(TRAINING_ANNOTATIONS_PATH)
    print(f"There are {len(distribution_map['0'])} images of level 0")
    for level in ['0']:
        num_images_in_level = len(distribution_map[level])
        num_supplementary_augmentations = 2500 - num_images_in_level
        num_augs_fulfilled = 0
        while num_augs_fulfilled < num_supplementary_augmentations:
            image = random.choice(distribution_map[level])
            img = cv2.imread(f"{DATASET_PATH}/processed_images/{image}")
            augs_to_apply = get_augs()
            distorted_image = apply_augmentations(augs_to_apply, img)
            distortion_string = "_".join(augs_to_apply)
            distorted_image_name = f"{image.removesuffix('.tif')}_{distortion_string}.tif"
            cv2.imwrite(
                f"{DATASET_PATH}/supplemented/{distorted_image_name}",
                distorted_image,
            )
            with open(TRAINING_ANNOTATIONS_PATH, "a", newline="") as file:
                writer = csv.DictWriter(file, ["image", "level"])
                row = {"image": distorted_image_name, "level": level}
                writer.writerow(row)
            num_augs_fulfilled += 1
    distribution_map = get_level_distribution_map(TRAINING_ANNOTATIONS_PATH)
    print(f"There are {len(distribution_map['0'])} images of level 0")
            


def horizontal_flip(img):
    return cv2.flip(img, 1)


def vertical_flip(img):
    return cv2.flip(img, 0)


def sp_noise(image):
    rand_prob = random.uniform(0.01, 0.035)
    print(rand_prob)
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < rand_prob:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def augment_data() -> None:
    with open(TRAINING_ANNOTATIONS_PATH, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in tqdm(annotations):
            if row["image"].endswith("PP.tif"):
                img = cv2.imread(f"{DATASET_PATH}/processed_images/{row['image']}")
            else:
                img = None
            if img is not None:
                # flipped = vertical_flip(img)
                # cv2.imwrite(
                #     f"{DATASET_PATH}/augmented_images/vflipped_images/{row['image'].removesuffix('.tif')}_vflipped.tif",
                #     flipped,
                # )
                with open(TRAINING_ANNOTATIONS_PATH, "a", newline="") as file:
                    writer = csv.DictWriter(file, row.keys())
                    row["image"] = f"{row['image'].removesuffix('.tif')}_vflipped.tif"
                    writer.writerow(row)


if __name__ == "__main__":
                
    supplementary_augmentations()      
    # apply_random_augmentation_combinations()
    # augment_data()

import csv
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm
from constants import DATASET_PATH
import cv2

TRAINING_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/trainLabels.csv")


def horizontal_flip(img):
    return cv2.flip(img, 1)


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
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
                # horizontally_flipped = horizontal_flip(img)
                # rotated = rotation(img, angle=15)
                speckled = sp_noise(img, 0.02)
                # cv2.imwrite(f"{DATASET_PATH}/augmented_images/{row['image'].removesuffix('.tif')}_flipped.tif", horizontally_flipped)
                # cv2.imwrite(f"{DATASET_PATH}/augmented_images/{row['image'].removesuffix('.tif')}_rotated.tif", rotated)
                cv2.imwrite(
                    f"{DATASET_PATH}/noisy_images/{row['image'].removesuffix('.tif')}_noisy.tif",
                    speckled,
                )
                with open(TRAINING_ANNOTATIONS_PATH, "a", newline="") as file:
                    writer = csv.DictWriter(file, row.keys())
                    row["image"] = f"{row['image'].removesuffix('.tif')}_noisy.tif"
                    writer.writerow(row)
                    # row['image'] = f"{row['image'].removesuffix('flipped.tif')}rotated.tif"
                    # writer.writerow(row)
                    # row['image'] = f"{row['image'].removesuffix('rotated.tif')}speckled.tif"
                    # writer.writerow(row)


if __name__ == "__main__":
    augment_data()

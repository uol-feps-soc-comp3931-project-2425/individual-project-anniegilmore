import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from utils import make_path

BASE_PATH = Path("../messidor-1")
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


def crop_image(path_to_images: Path, image):
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([1, 1, 1], dtype="uint8")
    image = cv2.imread(Path(f"{path_to_images}/{image}"))
    if image is None:
        print(f"Issue with image {image}")
        return None
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    _, thresholded = cv2.threshold(greyscale_image, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(output, contours, -1, 255, 3)
    region_of_interest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(region_of_interest)
    cv2.rectangle(thresholded, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped_image = image[y : y + h, x : x + w]
    return cropped_image


def denoise_image(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    return blurred


def histogram_equalization(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv_image)
    equalized_Y = cv2.equalizeHist(Y)
    equalized = cv2.merge([equalized_Y, U, V])
    converted = cv2.cvtColor(equalized, cv2.COLOR_YUV2BGR)
    return converted


def image_preprocessing(path_to_images: Path, new_path: Path) -> None:
    for image in tqdm(os.listdir(path_to_images)):
        cropped_image = crop_image(path_to_images, image)
        if cropped_image is None:
            continue
        blurred_image = denoise_image(cropped_image)
        equalized = histogram_equalization(blurred_image)
        cv2.imwrite(f"{new_path}/{image}", equalized)


if __name__ == "__main__":
    messidor_path = Path("../messidor-1/dataset_0.8/processed_images")
    make_path(messidor_path)
    for base in BASE_LIST:
        image_preprocessing(
            Path("../messidor-1/dataset_0.8/images"),
            Path("../messidor-1/dataset_0.8/processed_images"),
        )

    # make_path(Path("../aptos-2019-dataset/dataset_0.8/processed_images"))
    # image_preprocessing(Path("../aptos-2019-dataset/dataset_0.8/images"), Path("../aptos-2019-dataset/dataset_0.8/processed_images"))

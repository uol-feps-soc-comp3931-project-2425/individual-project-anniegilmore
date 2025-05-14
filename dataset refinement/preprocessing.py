import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from utils import make_path
from cv2.typing import MatLike


def apply_histogram_equalization(image: MatLike) -> MatLike:
    """Does histogram equalization on image to adjust brightness"""
    yuv_image: MatLike = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Y_channel, U_channel, V_channel = cv2.split(yuv_image)
    equalized_Y: MatLike = cv2.equalizeHist(Y_channel)
    equalized: MatLike = cv2.merge([equalized_Y, U_channel, V_channel])
    converted: MatLike = cv2.cvtColor(equalized, cv2.COLOR_YUV2BGR)
    return converted


def crop_image(image: MatLike) -> MatLike:
    """Implements Smart Cropping of image to eliminate black background"""
    lower: np.ndarray = np.array([0, 0, 0], dtype="uint8")
    upper: np.ndarray = np.array([1, 1, 1], dtype="uint8")
    greyscale_image: MatLike = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarisation_mask: MatLike = cv2.inRange(image, lower, upper)
    binarisation_output: MatLike = cv2.bitwise_and(image, image, mask=binarisation_mask)
    _, binarised_image = cv2.threshold(greyscale_image, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binarised_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(binarisation_output, contours, -1, 255, 3)
    largest_detected_shape: MatLike = max(contours, key=cv2.contourArea)
    x_coord, y_coord, width, height = cv2.boundingRect(largest_detected_shape)
    cv2.rectangle(
        binarised_image,
        (x_coord, y_coord),
        (x_coord + width, y_coord + height),
        (0, 255, 0),
        2,
    )
    cropped_image: MatLike = image[
        y_coord : y_coord + height, x_coord : x_coord + width
    ]
    return cropped_image


def run_preprocessing_pipeline(path_to_images: Path, new_path: Path) -> None:
    """Runs full processing path for every image in dataset, saving processed images"""
    make_path(new_path)
    for image in tqdm(os.listdir(path_to_images)):
        image: MatLike = cv2.imread(Path(f"{path_to_images}/{image}"))
        if image is None:
            print(f"Issue with image {image}")
            continue
        cropped_image: MatLike = crop_image(image)
        equalized: MatLike = apply_histogram_equalization(cropped_image)
        cv2.imwrite(f"{new_path}/{image}", equalized)
    return


if __name__ == "__main__":
    run_preprocessing_pipeline(
        Path("../messidor-1/dataset_0.7/images"),
        Path("../messidor-1/dataset_0.7/processed_images"),
    )

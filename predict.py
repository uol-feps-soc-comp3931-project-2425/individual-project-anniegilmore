from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
from numpy import uint8
import torch
from model_architecture import DiabeticRetinopathyNet
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import transforms
from tqdm import tqdm
from utils import get_device, make_path
from gradcam import GradCAM
from constants import DATA_PATH, DATASET_PATH
from torch.nn import Sequential
import numpy as np

PATH_TO_RESULTS: Path = Path(f"{DATASET_PATH}/predictions")


class PredictionDetails(BaseModel):
    """
    Stores information about the predictions made by the model
    """

    model: str
    total_images_scored: int
    time_taken: float
    predicted_grades: list[GradeDetails]


class GradeDetails(BaseModel):
    """
    Stores the actual predictions made by the model
    """

    image_name: str
    predicted_grade: int
    truth: int


def load_model(path_to_model: Path) -> DiabeticRetinopathyNet:
    """Initialise model to make predictions on data"""
    model: DiabeticRetinopathyNet = DiabeticRetinopathyNet(
        n_diabetic_retinopathy_levels=4
    )
    model.load_state_dict(
        torch.load(path_to_model, map_location=get_device()), strict=False
    )
    model.eval()
    return model


def load_ground_truth(path_to_labels: Path) -> list[dict]:
    """Load ground truth annotations for validation dataset"""
    predictions: list[dict] = []
    with open(path_to_labels, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in annotations:
            predictions.append({"image": row["image"], "truth": row["level"]})
    return predictions


def standardizing_transforms() -> transforms.Compose:
    """Define transforms to be applied to images in the validation dataset"""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


def get_heatmap(
    model: DiabeticRetinopathyNet, image: Image, image_tensor: torch.Tensor
) -> uint8:
    for param in model.custom_layer.parameters():
        param.requires_grad = True
    target_layer: Sequential = model.custom_layer[0]
    gradcam: GradCAM = GradCAM(model, target_layer)
    heatmap: np.ndarray = gradcam.generate_heatmap(image_tensor)
    overlayed_heatmap: uint8 = gradcam.overlay_heatmap(heatmap, image)
    return overlayed_heatmap


def predict_image(img_path: str, img_name: str, model: DiabeticRetinopathyNet) -> int:
    """Pass individual image through final model to get predicted DR level"""
    image: Image = Image.open(f"{img_path}/{img_name}").convert("RGB")
    transforms_to_apply: transforms.Compose = standardizing_transforms()
    image_tensor: torch.Tensor = transforms_to_apply(image).unsqueeze_(0)
    prediction = model(image_tensor.to(get_device()))
    grade_prediction: int = prediction["level"].argmax(dim=1).detach().cpu().numpy()
    heatmap_image: uint8 = get_heatmap(model, image, image_tensor)
    cv2.imwrite(
        f"{PATH_TO_RESULTS}/heatmaps/{img_name.removesuffix('.tif')}_gradcam.jpg",
        heatmap_image,
    )
    return grade_prediction


def get_predictions(
    path_to_model: Path, path_to_labels: Path, path_to_images: Path
) -> PredictionDetails:
    """Get predictions and prediction details for validation dataset"""
    start: float = time.time()
    model: DiabeticRetinopathyNet = load_model(path_to_model)
    images_to_predict: list[dict] = load_ground_truth(path_to_labels)
    prediction_details: PredictionDetails = PredictionDetails(
        model=path_to_model,
        total_images_scored=len(images_to_predict),
        time_taken=0.0,
        predicted_grades=[],
    )
    for image_to_predict in tqdm(images_to_predict):
        grade_prediction: int = predict_image(
            path_to_images, image_to_predict["image"], model
        )
        prediction_details.predicted_grades.append(
            GradeDetails(
                image_name=image_to_predict["image"],
                predicted_grade=grade_prediction,
                truth=image_to_predict["truth"],
            )
        )
    prediction_details.time_taken = time.time() - start
    return prediction_details


def predict_quality_of_images(
    path_to_model: Path, path_to_labels: Path, path_to_images: Path
) -> None:
    """Retrieve model predictions and heatmaps for validation dataset"""
    make_path(PATH_TO_RESULTS)
    prediction_details: PredictionDetails = get_predictions(
        path_to_model, path_to_labels, path_to_images
    )
    with open(f"{PATH_TO_RESULTS}/results.json", "w", newline="") as f:
        json_str = prediction_details.json(indent=2)
        f.write(json_str)
        f.flush()
    return


if __name__ == "__main__":
    path_to_model = f"{DATA_PATH}Iteration_69/checkpoints/checkpoint-epoch-18.pth"
    path_to_labels = Path(f"{DATASET_PATH}/validateLabels.csv")
    path_to_images = Path(f"{DATASET_PATH}/images")
    predict_quality_of_images(path_to_model, path_to_labels, path_to_images)

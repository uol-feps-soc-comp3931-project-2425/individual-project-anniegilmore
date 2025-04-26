import csv
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from model_architecture import DiabeticRetinopathyNet
from constants import DATA_PATH, DATASET_PATH, ITERATION
from utils import get_device
import os
import torch.nn.functional as F
from gradcam import GradCAM
from tqdm import tqdm
import json

VALIDATION_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/validateLabels.csv")
IMAGES_TO_PREDICT = Path(f"{DATASET_PATH}/images")

def load_model(model_name: Path) -> DiabeticRetinopathyNet:
    model = DiabeticRetinopathyNet(n_diabetic_retinopathy_levels = 4)
    path_to_model = f"{DATA_PATH}/{ITERATION.replace(' ', '_')}/checkpoints/{model_name}"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()  # Switch to evaluation mode

    device = get_device()
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    return model
    
def load_ground_truth(path_to_labels: Path) -> dict:
    predictions: dict = {}
    with open(path_to_labels, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in annotations:
            predictions[row['image']] = {'truth': row['level'], 'prediction': None}
    return predictions

def predict_grades(model: DiabeticRetinopathyNet, device: torch.device) -> dict:
    # Load and transform the image
    predictions: dict = load_ground_truth(VALIDATION_ANNOTATIONS_PATH)
    for image_to_predict in tqdm(predictions):
        image = Image.open(f"{IMAGES_TO_PREDICT}/{image_to_predict}").convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension: [1, C, H, W]
        img_tensor = img_tensor.to(device)  # Send to same device as model

        output = model(img_tensor)
        prediction = output["level"].argmax(dim=1).detach().cpu().numpy()
        # print(f"Model made a prediction of {prediction[0]}")
        predictions[image_to_predict]['prediction'] = prediction[0]
        for param in model.conv_head.parameters():
            param.requires_grad = True
        target_layer = model.conv_head[8]

        gradcam = GradCAM(model, target_layer)

        # Generate GradCAM heatmap
        heatmap = gradcam.generate(img_tensor)

        # Overlay and visualize
        overlay_img = gradcam.overlay(heatmap, image)
        cv2.imwrite(f"{DATASET_PATH}/predictions/{image_to_predict.removesuffix('.tif')}_gradcam.jpg", overlay_img)

            
    return predictions

def analyse_results(predictions: dict) -> None:
    correct_preds: list = []
    for image in predictions.keys():
        if predictions[image]['prediction'] == predictions[image]['truth']:
            correct_preds.append(image)
    
    accuracy = len(correct_preds) / len(predictions)
    print(correct_preds)
    print(f"reached accuracy of {accuracy}")
    print(predictions)
    # with open('result.json', 'w') as fp:
    #     json.dump(predictions, fp)
        


def show_gradcam_on_image(img_tensor, gradcam, img_name):
    img = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img - img.min()
    img = img / img.max()
    
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + img
    cam = cam / cam.max()

    plt.imshow(cam)
    plt.axis("off")
    plt.savefig(f"{DATASET_PATH}/predictions/{img_name}_gradcam.jpeg")
    plt.clf()
    plt.close()
    
    
def try_predict(model_path: Path):
    model = load_model(model_path)
    predictions = predict_grades(model, get_device())
    analyse_results(predictions)
            


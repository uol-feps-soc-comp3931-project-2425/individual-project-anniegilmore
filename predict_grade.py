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
import torch.nn.functional as F

VALIDATION_ANNOTATIONS_PATH = Path(f"{DATASET_PATH}/validateLabels.csv")
IMAGES_TO_PREDICT = Path(f"{DATASET_PATH}/images")

def load_model(model_name: Path) -> DiabeticRetinopathyNet:
    model = DiabeticRetinopathyNet()
    path_to_model = f"{DATA_PATH}/{ITERATION.replace(" ", "_")}/checkpoints/{model_name}"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()  # Switch to evaluation mode

    device = get_device()
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    
def load_ground_truth(path_to_labels: Path) -> list[dict]:
    predictions: list[dict] = []
    with open(path_to_labels, "r", newline="") as file:
        annotations: csv.DictReader[str] = csv.DictReader(file)
        for row in annotations:
            predictions.append({'image': row['image'], 'truth': row['label'], 'prediction': None})
    return predictions

def predict_grades(model: DiabeticRetinopathyNet, device: torch.device) -> dict:
    # Load and transform the image
    predictions: list[dict] = load_ground_truth(VALIDATION_ANNOTATIONS_PATH)
    for image_to_predict in predictions:
        image = Image.open(f"{IMAGES_TO_PREDICT}/{image_to_predict['image']}").convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension: [1, C, H, W]
        img_tensor = img_tensor.to(device)  # Send to same device as model
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1)
            predictions['prediction'] = pred
            gradients = model.feature_map.grad  # This will be None unless we register a hook
            activations = model.feature_map
            
            gradients = []
            def save_gradient(grad):
                gradients.append(grad)

            hook = model.feature_map.register_hook(save_gradient)
            output = model(img_tensor)
            gradients = gradients[0].squeeze(0)  
            activations = model.feature_map.squeeze(0)  # [C, H, W]

            # Compute Grad-CAM: weights = global average pooling over gradients
            weights = gradients.mean(dim=(1, 2))  # [C]
            gradcam = torch.zeros(activations.shape[1:], dtype=torch.float32)
            
            for i, w in enumerate(weights):
                gradcam += w * activations[i]

            gradcam = F.relu(gradcam)
            gradcam = gradcam - gradcam.min()
            gradcam = gradcam / gradcam.max()
            gradcam = gradcam.detach().cpu().numpy()

            # Resize to input image size
            gradcam = cv2.resize(gradcam, (img_tensor.shape[3], img_tensor.shape[2]))
            show_gradcam_on_image(img_tensor, gradcam, image_to_predict['image'])
            
    return predictions


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


import csv
from pathlib import Path
from typing import Any


import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from constants import DATASET_PATH
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


PATH_TO_IMAGES = Path(f"{DATASET_PATH}/processed_images")


class AttributesDataset:
    def __init__(self) -> None:

        self.diabetic_retinopathy_levels = ["0", "1", "2"]
        # self.diabetic_retinopathy_levels = ["airplane", "cat", "deer"]
        self.num_classes = len(self.diabetic_retinopathy_levels)

        self.level_to_id = dict(
            zip(
                self.diabetic_retinopathy_levels,
                range(len(self.diabetic_retinopathy_levels)),
                strict=False,
            )
        )


class RetinalImageDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        attributes: AttributesDataset,
        transform: transforms.Compose,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.attr = attributes
        # arrays to store ground truth labels
        self.data = []
        self.diabetic_retinopathy_levels = []
        self.class_weights = []
        
        class_counts = {}
        for potential_level in attributes.diabetic_retinopathy_levels:
            class_counts[potential_level] = 0

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row["image"])
                self.diabetic_retinopathy_levels.append(
                    self.attr.level_to_id[row["level"]]
                )
                class_counts[row["level"]] += 1
        
        total_samples = len(self.data)
        print(f"There are {total_samples} images")
        for count in class_counts:
            weight = 1 / (class_counts[count] / total_samples)
            self.class_weights.append(weight)
            

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path: str = f"{self.data[idx]}.jpg"
        # img_path: str = f"{self.data[idx]}"
        img: Image.Image = Image.open(f"{PATH_TO_IMAGES}/{img_path}").convert("RGB")
        if self.transform:
            img = self.transform(img)
        dict_data = {
            "img": img,
            "levels": self.diabetic_retinopathy_levels[idx],
        }
        return dict_data

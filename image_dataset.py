import csv
from pathlib import Path
from typing import Any


import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from constants import DATASET_PATH

# defining constant paths to datsets
PATH_TO_IMAGES = Path(f"{DATASET_PATH}/images")
PATH_TO_SUPPLEMENTED_IMAGES = Path(f"{DATASET_PATH}/supplemented")


class AttributesDataset:
    """
    Defines possible DR levels that can be predicted
    """

    def __init__(self) -> None:
        """
        Initialises AttributesDataset object
        """
        self.diabetic_retinopathy_levels = ["0", "1", "2", "3"]
        self.num_classes = len(self.diabetic_retinopathy_levels)

        self.level_to_id = dict(
            zip(
                self.diabetic_retinopathy_levels,
                range(len(self.diabetic_retinopathy_levels)),
                strict=False,
            )
        )


class RetinalImageDataset(Dataset):
    """Holds images and corresponding ground truth labels for DR diagnosis datasets"""

    def __init__(
        self,
        annotation_path: Path,
        attributes: AttributesDataset,
        transform: transforms.Compose,
    ) -> None:
        """
        Initialises RetinalImageDataset with path to ground truth labels file and AttributesDataset

        Args:
            annotation_path (Path): path to file containing ground truth labels
            attributes (AttributesDataset): AttributesDataset defining possible DR levels of data
            transform (transforms.Compose): transforms to be applied to images in the dataset
        """
        super().__init__()
        self.transform = transform
        self.attr = attributes
        self.data = []
        self.diabetic_retinopathy_levels = []

        with open(annotation_path) as f:
            reader: csv.DictReader = csv.DictReader(f)
            for row in reader:
                self.data.append(row["image"])
                self.diabetic_retinopathy_levels.append(
                    self.attr.level_to_id[row["level"]]
                )

    def __len__(self) -> int:
        """Gets number of items in dataset"""
        return len(self.data)

    def __getitem__(self, data_index: int) -> dict[str, Any]:
        """Loads and transforms individual image in dataset"""
        img_path: str = f"{self.data[data_index]}"
        if img_path.endswith("PP.tif"):
            img: Image.Image = Image.open(f"{PATH_TO_IMAGES}/{img_path}").convert("RGB")
        else:
            img: Image.Image = Image.open(
                f"{PATH_TO_SUPPLEMENTED_IMAGES}/{img_path}"
            ).convert("RGB")
        if img is None:
            print(f"Error loading image {img_path}")
        if self.transform:
            img = self.transform(img)
        dict_data = {
            "img": img,
            "levels": self.diabetic_retinopathy_levels[data_index],
        }
        return dict_data

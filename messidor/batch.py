import os
from pathlib import Path
import shutil


BASE_PATH = Path("../messidor-1/dataset_0.8/processed_images")
DIR_1 = Path("../messidor-1/dataset_0.8/batch_1")
DIR_2 = Path("../messidor-1/dataset_0.8/batch_2")
DIR_3 = Path("../messidor-1/dataset_0.8/batch_3")
DIR_4 = Path("../messidor-1/dataset_0.8/batch_4")
DIR_5 = Path("../messidor-1/dataset_0.8/batch_5")
DIR_6 = Path("../messidor-1/dataset_0.8/batch_6")
DIR_7 = Path("../messidor-1/dataset_0.8/batch_7")
DIR_8 = Path("../messidor-1/dataset_0.8/batch_8")
DIR_9 = Path("../messidor-1/dataset_0.8/batch_9")
DIR_10 = Path("../messidor-1/dataset_0.8/batch_10")

for path in [DIR_1, DIR_2, DIR_3, DIR_4, DIR_5, DIR_6, DIR_7, DIR_8, DIR_9, DIR_10]:
    Path.mkdir(path, parents=True, exist_ok=True)

for count, image in enumerate(os.listdir(BASE_PATH)):
    if count < 120:
        shutil.copy(BASE_PATH / image, DIR_1)
    if count >= 120 and count < 240:
        shutil.copy(BASE_PATH / image, DIR_2)
    if count >= 240 and count < 360:
        shutil.copy(BASE_PATH / image, DIR_3)
    if count >= 360 and count < 480:
        shutil.copy(BASE_PATH / image, DIR_4)
    if count >= 480 and count < 600:
        shutil.copy(BASE_PATH / image, DIR_5)
    if count >= 600 and count < 720:
        shutil.copy(BASE_PATH / image, DIR_6)
    if count >= 720 and count < 840:
        shutil.copy(BASE_PATH / image, DIR_7)
    if count >= 840 and count < 960:
        shutil.copy(BASE_PATH / image, DIR_8)
    if count >= 960 and count < 1080:
        shutil.copy(BASE_PATH / image, DIR_9)
    if count >= 1080:
        shutil.copy(BASE_PATH / image, DIR_10)

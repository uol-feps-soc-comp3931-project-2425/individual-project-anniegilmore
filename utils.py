import logging
from pathlib import Path
import torch


def make_path(path_to_create: Path) -> None:
    Path.mkdir(path_to_create, parents=True, exist_ok=True)


def setup_logger(
    name: str, log_file: Path, level: int = logging.INFO
) -> logging.Logger:
    make_path(log_file.parent)
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

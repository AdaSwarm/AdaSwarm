"""Parameters for the model training"""
import os
import torch
from torch import cuda

DEVICE = torch.device("cuda:0" if cuda.is_available() else "cpu")


def get_device():
    """Obtain the processor type (CPU or GPU)

    Returns:
        torch.device: Available device
    """
    return DEVICE


def write_batch_frequency():
    """Get write batch frequency from environment variable

    Returns:
        [int]: Frequency of writes to Tensorbaord
    """
    return int(os.environ.get("ADASWARM_WRITE_BATCH_FREQUENCY", "50"))


def number_of_epochs() -> int:
    """Set the number of epochs to run
    Returns:
        [int]: Number of epochs
    """
    return int(os.environ.get("ADASWARM_NUMBER_OF_EPOCHS", "40"))


def dataset_name() -> str:
    """Set the dataset name
    Returns:
        [str]: Name of dataset
    """
    return os.environ.get("ADASWARM_DATASET_NAME", "Iris")


def log_level() -> str:
    """Set the default log level
    Returns:
        [str]: Log level
    """
    return os.environ.get("LOGLEVEL", "INFO").upper()

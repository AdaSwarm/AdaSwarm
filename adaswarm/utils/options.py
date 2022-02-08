import os
from adaswarm.utils.strings import str_to_bool


def is_adaswarm():
    """Determine whether or not to run with AdaSwarm optimiser

    Returns:
        bool: True if wanting to run with AdaSwarm, False for Adam
    """
    return str_to_bool(os.environ.get("USE_ADASWARM", "True"))


def write_batch_frequency():
    """Get write batch frequency from environment variable

    Returns:
        [int]: Frequency of writes to Tensorbaord
    """
    return int(os.environ.get("ADASWARM_WRITE_BATCH_FREQUENCY", "50"))


def write_to_tensorboard(batch_idx: int) -> bool:
    """Boolean to determine whether to write to tensorboard

    Args:
        batch_idx (int): batch iteration number

    Returns:
        bool: boolean flag, True to write
    """
    frequency = write_batch_frequency()
    return batch_idx % frequency == (frequency - 1)


def get_tensorboard_log_path(run_type: str) -> str:
    """Obtain the folder into which to save the tensorboard writer files
    Inputs:
        [str]: run type i.e. train/eval
    Returns:
        [str]: relative path tailored to experiment
    """
    tensorboard_dir = "runs_adaswarm" if is_adaswarm() else "runs_adam"

    return os.path.join("mnist_performance", tensorboard_dir, run_type)


def number_of_epochs() -> int:
    """Set the number of epochs to run
    Returns:
        [int]: Number of epochs
    """
    return int(os.environ.get("ADASWARM_NUMBER_OF_EPOCHS", "200"))


def dataset_name() -> str:
    """Set the dataset name
    Returns:
        [str]: Name of dataset
    """
    return os.environ.get("ADASWARM_DATASET_NAME", "MNIST")

import os
from adaswarm.utils.strings import str_to_bool


def is_adaswarm():
    """Determine whether or not to run with AdaSwarm optimiser

    Returns:
        bool: True if wanting to run with AdaSwarm, False for Adam
    """
    return str_to_bool(os.environ.get("USE_ADASWARM", "True"))

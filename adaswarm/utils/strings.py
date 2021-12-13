def str_to_bool(string_in: str):
    """convert a string to boolean

    Args:
        string_in (string): Text from config file value

    Raises:
        ValueError: Cannot convert the string to a bool as it isn't True or False

    Returns:
        Boolean: True/False value
    """
    if string_in.lower() == "true":
        return True
    elif string_in.lower() == "false":
        return False
    else:
        raise ValueError(f"Cannot convert {string_in} to a bool")

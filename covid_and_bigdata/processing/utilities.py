from datetime import datetime


def change_timestamp_to_month(x: float):
    """
    Parameters
    ----------
    x: `float`, required
        The float value of the timestamp

    Returns
    ----------
    The corresponding month (1 - 10) in the year 2020
    """
    for i in range(1, 11):
        if x >= datetime(year=2020, month=i, day=1).timestamp() and x < datetime(year=2020, month=i+1, day=1).timestamp():
            return i


"""
    Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    api func for gap views
"""
from pandas import Series


def count_gaps(data: Series, gap_marker: float = -1.0) -> int:
    """ count gaps in data """
    cnt = 0
    for val in data.values:
        if val == gap_marker:
            cnt += 1
    return cnt

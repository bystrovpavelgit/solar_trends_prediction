""" api for stat views """
from webapp.config import VALID_VALUES
from webapp.utils.dataframe_util import get_enriched_dataframe
from webapp.utils.trends_util import exponential_smoothing, \
    double_exponential_smoothing, hw_exponential_smoothing


def get_smoothed_data_by_type(smooth_type: str) -> tuple:
    """ get smoothed data according to type """
    if smooth_type is None:
        raise ValueError("smooth_type is empty")
    data = get_enriched_dataframe()
    time = data["year_float"].values.tolist()
    sunspots = data["sunspots"].values.tolist()
    if smooth_type == VALID_VALUES[1]:
        smoothed = data["mean_12y"].values.tolist()
    elif smooth_type == VALID_VALUES[2]:
        smoothed = exponential_smoothing(data["sunspots"], .25)
    elif smooth_type == VALID_VALUES[3]:
        smoothed = double_exponential_smoothing(data["sunspots"], .2, .2)
    elif smooth_type == VALID_VALUES[4]:
        smoothed = hw_exponential_smoothing(data.sunspots)
    elif smooth_type == VALID_VALUES[0]:
        smoothed = data["mean_3y"].values.tolist()
    else:
        return [], [], []
    return time, sunspots, smoothed

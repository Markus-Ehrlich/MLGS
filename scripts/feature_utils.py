"""Helper functions for feature engineering on weather data."""

import numpy as np

def calculate_daily_trend(st_hourly_values):
    """
    Calculates the trend (slope) of hourly values over a day.
    """
    y_values = st_hourly_values.values  # e.g. 24 hourly temperature values
    x_hours = np.arange(len(y_values))  # hours 0 to 23
    slope = np.polyfit(x_hours, y_values, 1)[0]  # m of the line y = m*x + b
    return slope

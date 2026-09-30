"""
------------------------- Helper functions for the pipeline object ------------------------------
---------------------------- also holds encoder helper functions -----------------------
"""
import calendar
import hashlib
from datetime import datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from typing import Optional

import numpy as np
import pandas as pd
from dateutil import parser


# --------- Time helpers ------------
def convert_timestamp(timestamp: [str | datetime]) -> datetime:
    """
    Confirm / convert time object
    Parameters
    ----------
    timestamp : convert a string into a pd.timestamp or datetime obj

    Returns
    -------
    time-style object
    """
    if isinstance(timestamp, datetime):
        return timestamp
    elif isinstance(timestamp, pd.Timestamp):
        return timestamp.to_pydatetime()
    elif isinstance(timestamp, str) or (isinstance(timestamp, float)):
        try:
            return pd.Timestamp(timestamp).to_pydatetime()
        except ValueError:
            return parser.parse(timestamp)
    return timestamp


def minute_per_hour(timestamp) -> float:
    return (2 * np.pi * timestamp.minute) / 60


def inv_minute_per_hour(decimal: float) -> float:
    return decimal * 60


def hour_per_day(timestamp) -> float:
    if timestamp.minute >= 30:
        _hour = timestamp.hour + 1
    else:
        _hour = timestamp.hour
    return (_hour / 24) * (2 * np.pi)


def inv_hour_per_day(decimal: float) -> float:
    return decimal * 24


def day_per_week(timestamp) -> float:
    return (timestamp.isoweekday() / 7) * (2 * np.pi)


def inv_day_per_week(decimal: float) -> float:
    return decimal * 7


def day_per_month(timestamp) -> float:
    days_in_month = calendar.monthrange(timestamp.year, timestamp.month)[1]
    return (timestamp.day / days_in_month) * (2 * np.pi)


def inv_day_per_month(decimal: float) -> float:
    return decimal * 30.437


def day_per_year(timestamp) -> float:
    day_of_year = (timestamp - datetime(timestamp.year, 1, 1)).days + 1
    is_leap_year = calendar.isleap(timestamp.year)
    days_in_year = 366 if is_leap_year else 365
    return (day_of_year / days_in_year) * (2 * np.pi)


def inv_day_per_year(decimal: float) -> float:
    return decimal * 365.25


def week_per_month(timestamp) -> float:
    days_in_month = calendar.monthrange(timestamp.year, timestamp.month)[1]

    return (2 * np.pi * round(timestamp.day / 7, 1)) / (days_in_month // 7)


def inv_week_per_month(decimal: float) -> float:
    return decimal * 4.345


def week_per_year(timestamp) -> float:
    day_of_year = (timestamp - datetime(timestamp.year, 1, 1)).days + 1
    return ((round(day_of_year / 7), 1) / 52.1775) * (2 * np.pi)


def inv_week_per_year(decimal: float) -> float:
    return decimal * 52.1775


def month_per_year(timestamp) -> float:
    return (timestamp.month / 12) * (2 * np.pi)


def inv_month_per_year(decimal: float) -> float:
    return decimal * 12


def localize_timestamp(timestamp: datetime, timezone_id: int) -> datetime:
    """
    Localizes a timestamp to a given timezone id (gotten from usersvc
    Parameters
    ----------
    timestamp : timestamp object (ideally already a datetime object, but we'll convert it
    timezone_id : localize the timestamp to a timezone

    Returns
    -------
    the modified datetime object to correctly reflect the user's timezone.  This is a simplified version that doesn't
        take into account daylight savings, so it may be +/- 1 hr
    """
    # mask NaNs
    _timeobj = convert_timestamp(timestamp)

    if (_timeobj == np.nan) or isinstance(_timeobj, float):
        return timestamp

    _timeobj = convert_timestamp(timestamp)
    modified_timestamp = _timeobj + timedelta(hours=float(timezone_id))

    return modified_timestamp


# ------------------------- dispatcher lookups -------------------------
DISPATCHER = {
    "minute/hour": minute_per_hour,
    "hour/day": hour_per_day,
    "day/week": day_per_week,
    "day/month": day_per_month,
    "day/year": day_per_year,
    "week/month": week_per_month,
    "week/year": week_per_year,
    "month/year": month_per_year,
}
INVERT_DISPATCHER = {
    "minute/hour": inv_minute_per_hour,
    "hour/day": inv_hour_per_day,
    "day/week": inv_day_per_week,
    "day/month": inv_day_per_month,
    "day/year": inv_day_per_year,
    "week/month": inv_week_per_month,
    "week/year": inv_week_per_year,
    "month/year": inv_month_per_year,
}


# --------- Data Distribution Helpers / Outliers ------------
def calcualte_iqr_bounds(values, extension: float = 2.0) -> tuple[float, float]:
    """
    Calculates the inter-quartile range for the given distribution
    Parameters
    ----------
    values : iterable / array-like sample values
    extension : how far the IQR reach sits - standard is 1.5; 1.7 gives us 3 standard deviations

    Returns
    -------
    the lower and upper bounds for these samples
    """
    _vs = values.dropna()
    q1 = np.quantile(values, 0.25, method="midpoint")
    q3 = np.quantile(values, 0.75, method="midpoint")

    # 1.7 gives us 3 standard deviations, but 1.5 is common prac
    extended_iqr = abs((q3 - q1) * extension)
    iqr_lower_bound = float(q1 - extended_iqr)
    iqr_upper_bound = float(q3 + extended_iqr)
    return iqr_lower_bound, iqr_upper_bound


def find_outliers(
        values, idxs_to_impute: set, bounds: Optional[tuple] = None
) -> tuple[list[int,], tuple[float, float]]:
    """
    Use the inter-quartile rule to find outliers - any responses outside the extended IQR range is ruled an outlier.
    Parameters
    ----------
    values : sample values
    idxs_to_impute : indices which have been already determined as outliers
    bounds : tuple of lower and upper bounds (if already calculated)

    Returns
    -------
    list of the indices that fall outside the bounds;
    """
    if bounds is None:
        iqr_lower_bound, iqr_upper_bound = calcualte_iqr_bounds(values)
    else:
        iqr_lower_bound, iqr_upper_bound = bounds

    idxs_to_impute.update(np.where(values > iqr_upper_bound)[0].tolist())
    idxs_to_impute.update(np.where(values < iqr_lower_bound)[0].tolist())

    return list(idxs_to_impute), (iqr_lower_bound, iqr_upper_bound)


# -------------------- text helpers -------------------------
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.reset()
        self.fed = []

    def handle_data(self, d: str) -> None:
        self.fed.append(d)

    def handle_entityref(self, name: str) -> None:
        self.fed.append("&%s;" % name)

    def handle_charref(self, name: str) -> None:
        self.fed.append("&#%s;" % name)

    def get_data(self) -> str:
        return "".join(self.fed)


def _strip_once(value: str) -> str:
    """
    Internal tag stripping func used by HTML Parser and strip_tags()
    Parameters
    ----------
    value : str, string to process for html tags

    Returns
    -------
    str, with one pass of html tag stripped
    """
    s = MLStripper()
    s.feed(value)
    s.close()
    return s.get_data()


def strip_tags(value: str) -> str:
    """
    Return the given HTML with all tags stripped.
    # Note: in typical case this loop executes _strip_once once. Loop condition
    # is redundant, but helps to reduce number of executions of _strip_once.
    Parameters
    ----------
    value : str, the string to remove html tags

    Returns
    -------
    str, html stripped string
    """
    value = unescape(str(value))
    while "<" in value and ">" in value:
        new_value = _strip_once(value)
        if value.count("<") == new_value.count("<"):
            # _strip_once wasn't able to detect more tags.
            break
        value = new_value
    return value


def hash_string(value: str) -> str:
    """
    Hash a string using SHA256 - this ONE-WAY HASH should match seceng's implementation used in the anonymization
        pipeline
    To be used to bring parity between training hashed data and production data. (only use with prediction data,
        not at train!)

    Parameters
    ----------
    value : str, the string to hash

    Returns
    -------
    str of the hashed value
    """
    if not isinstance(value, str):
        value = str(value)

    hash_object = hashlib.sha256(value.encode("utf-8"))

    return hash_object.hexdigest()

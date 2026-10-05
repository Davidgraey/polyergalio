"""
------------------------ Chronological preprocessing / encoders ------------------------------
-------------------- includes cyclic, relative, and "time lag"encoding-----------------------
"""

from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from polyergalio.encoders.encoder_constants import DELTA_LOOKUP, Period
from polyergalio.encoders.encoder_utils import (
    DISPATCHER,
    INVERT_DISPATCHER,
    convert_timestamp,
    localize_timestamp,
)
from polyergalio.encoders.encoders import Processor


# ------------------------- Time Processors -------------------------
class TimeCycleProcessor(Processor):
    def __init__(
        self,
        target: str,
        numerator: Period = Period.DAY,
        denominator: Period = Period.WEEK,
    ):
        """
        Cyclic processor - we will encode time values as a ratio, and transform that into sine / cosine
            representations. Each target variable will have two output vectors, one for sine, one for cosine.
            This should let us encode timestamps as cyclic; day of the week, month of the year, etc.
        Parameters
        ----------
        target : The column name of the intended variable
        numerator : the numerator, as a PERIOD enum. (period.Day)
        denominator : the denominator of our cyclic ratio (period.Week)
        """
        super().__init__(target)
        self.numerator: Period = numerator
        self.denominator: Period = denominator
        self._func = DISPATCHER[f"{numerator.value}/{denominator.value}"]
        self._inv_func = np.vectorize(INVERT_DISPATCHER[f"{numerator.value}/{denominator.value}"])

    def fit(self, values: pd.Series) -> bool:
        """
        Fit the processor using the intended variable's values
        Parameters
        ----------
        values : iterable, series or array - datapoints of timestamps or timestamp-like strings to process

        Returns
        -------
        True if fitting was successful
        """
        self.fitted = True
        return True

    def encode(self, values: pd.Series):
        _values = values.dropna(axis=0)
        _vals = _values.apply(convert_timestamp).apply(self._func)

        sin = np.sin(_vals)
        cos = np.cos(_vals)

        if self.obs_min_max is None:
            self.obs_min_max = (
                (np.nanmin(sin), np.nanmax(sin)),
                (np.nanmin(cos), np.nanmax(cos)),
            )
        return {
            f"{self.target}_{self.numerator.value}_{self.denominator.value}_sin": sin,
            f"{self.target}_{self.numerator.value}_{self.denominator.value}_cos": cos,
        }

    def fit_encode(self, values: pd.Series):
        success = self.fit(values)
        return self.encode(values)

    def inverse(self, values: pd.Series):
        s = self._inv_func(np.arcsin(values)) / (2 * np.pi)
        c = self._inv_func(np.arccos(values)) / (2 * np.pi)
        return s, c

    @property
    def metadata(self):
        return {
            str(self.target): {
                "period": f"{self.numerator} | {self.denominator}",
                "output_dimension": 2,
                "output_type": "float",
                "enc_type": "chronological",
                "variable_names": [
                    f"{self.target}_{self.numerator.value}_{self.denominator.value}_sin",
                    f"{self.target}_{self.numerator.value}_{self.denominator.value}_cos",
                ],
            }
        }


class TimeAbsoluteProcessor(Processor):
    state_names = ("fitted_min", "fitted_max")

    def __init__(
        self,
        target: str,
        primary_interval: Period = Period.DAY,
    ):
        """
        Encodes time as a value from anchor -> timestamp.
        Connect it to three columns, in order: target (end date), anchor (start date) and timezone.
        Constrains to values 0-1
        Parameters
        ----------
        target : The column name of the intended variable
        primary_interval : interval that we want the absolute value of the different between anchor and target (start and end)
        """
        super().__init__(target)

        self.primary_interval = primary_interval
        self._localize: callable = np.vectorize(localize_timestamp)
        self.fitted_min = None
        self.fitted_max = None

    def get_localized_deltas(
        self, target_values: pd.Series, anchor_values: pd.Series, timezone_values: pd.Series
    ) -> tuple[NDArray, NDArray]:
        """
        localize the times using the user's account level timezone, then determine the lag between anchor and target
        Parameters
        ----------
        target_values : t2, the changed value to compare against the anchor value
        anchor_values : t1, the base values to compare (as datetime obj or pandas timestamp)
        timezone_values : series of timezones for each user's account

        Returns
        -------
        the difference between anchor and target in self.primary_interval, and the valid indices (where we don't have NaNs)
        """
        valid_idx = ~target_values.isna() & ~anchor_values.isna()

        local_target = self._localize(target_values, timezone_values)
        local_anchor = self._localize(anchor_values, timezone_values)

        count = np.empty_like(local_target)
        delta = np.abs(local_target[valid_idx] - local_anchor[valid_idx]).flatten()
        lookup = DELTA_LOOKUP[self.primary_interval.value]
        count[valid_idx] = np.array([one_delta.seconds / lookup for one_delta in delta])

        return count, valid_idx

    def fit(self, target_values: pd.Series, anchor_values: pd.Series, timezone_values: pd.Series):
        values, valid_idx = self.get_localized_deltas(target_values, anchor_values, timezone_values)
        self.fitted_min = np.nanmin(values[valid_idx])
        self.fitted_max = np.nanmax(values[valid_idx])
        self.fitted = True
        return True

    def encode(self,
               target_values: pd.Series,
               anchor_values: pd.Series,
               timezone_values: pd.Series) -> dict:
        values, valid_idx = self.get_localized_deltas(target_values, anchor_values, timezone_values)

        # clip any values to 25% +/- our fitted max and min. This should help control drift a little.
        values[valid_idx] = np.clip(
            values[valid_idx], a_min=0.75 * self.fitted_min, a_max=1.25 * self.fitted_max
        )

        values[valid_idx] = (values[valid_idx] - self.fitted_max) / (self.fitted_max - self.fitted_min)
        values = values.astype(float)

        if self.obs_min_max is None:
            self.obs_min_max = (np.nanmin(values), np.nanmax(values))
        return {f"{self.target}_delta": values}

    def fit_encode(self, target_values: pd.Series, anchor_values: pd.Series, timezone_values: pd.Series):
        self.fit(target_values, anchor_values, timezone_values)
        return self.encode(target_values, anchor_values, timezone_values)

    def inverse(self, values):
        return values * ((self.fitted_max - self.fitted_min) + self.fitted_min)

    @property
    def metadata(self):
        return {
            str(self.target): {
                "period": f"{self.primary_interval}",
                "output_dimension": 1,
                "enc_type": "chronological",
                "output_type": "float",
                "variable_names": [f"{self.target}_delta"],
            }
        }

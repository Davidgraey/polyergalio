"""
------------------------ Numeric preprocessing / encoders ------------------------------
-------------------- includes outlier clipping, standardize and normalize-----------------------
"""

from typing import Optional

import numpy as np
import pandas as pd

from polyergalio.encoders.encoder_utils import calcualte_iqr_bounds, find_outliers
from polyergalio.encoders.encoders import Processor


class NuemricNormalizeProcessor(Processor):
    _structural_state_keys = Processor._structural_state_keys + (
        "fitted_min", "fitted_max", "seen_samples", "bounds",
    )

    def __init__(
        self,
        target: str,
        impute_method: str = "clip",
        encoding_range: tuple[float, float] = (0.0, 1.0),
        impute_outliers: bool = False,
    ):
        """
        Normalize all values between args min and max (eg 0 and 1 or -1 and 1)
        Parameters
        ----------
        target : the name of the target column
        impute_method : method to use for imputating outliers
        encoding_range : low, high values to bind the values to - eg 0, 1 or -1, 1
        impute_outliers : to impute or not impute
        """
        super().__init__(target)

        self.encoding_range = encoding_range
        self.fitted_min = 1
        self.fitted_max = 1
        self.impute_outliers: bool = impute_outliers
        self.impute_method = impute_method.lower()
        self.seen_samples = 0
        self.bounds = ()

    def fit(self, values: iter) -> bool:
        """
        Fits the encoder on the given values. There are no checks or validations at this level, it relies on the
        pipeline object to validate values, data formats, etc.
        Parameters
        ----------
        values : a dataframe, series or other iterable object

        Returns
        -------
        True if successful
        """
        if isinstance(values, dict) or isinstance(values, pd.DataFrame):
            _v = values.get(self.target).dropna(axis=0)
        else:
            _v = values

        if self.impute_outliers:
            self.bounds = calcualte_iqr_bounds(values, extension=2.0)
            idxs_to_impute, self.bounds = find_outliers(
                values=_v, idxs_to_impute=set(), bounds=self.bounds
            )
            if idxs_to_impute:
                if self.impute_outliers == "clip":
                    _v[idxs_to_impute] = np.clip(
                        _v[idxs_to_impute], a_min=self.bounds[0], a_max=self.bounds[1]
                    )
                elif self.impute_method == "mean":
                    _v[idxs_to_impute] = np.nanmean(_v)
                elif self.impute_method == "median":
                    _v[idxs_to_impute] = np.nanmedian(_v)

        if self.seen_samples == 0:
            self.fitted_min = float(np.nanmin(_v))
            self.fitted_max = float(np.nanmax(_v))
        else:
            self.fitted_min = float(min(self.fitted_min, np.nanmin(_v)))
            self.fitted_max = float(max(self.fitted_max, np.nanmax(_v)))
        self.seen_samples += len(_v)
        self._fitted = True
        return True

    def encode(self, values):
        if isinstance(values, dict) or isinstance(values, pd.DataFrame):
            _v = values.get(self.target).dropna(axis=0)
        else:
            _v = values
        # --- outlier removal ---
        idxs_to_impute = set()

        if self.impute_outliers:
            idxs_to_impute, self.bounds = find_outliers(
                values=_v, idxs_to_impute=idxs_to_impute, bounds=self.bounds
            )

        if idxs_to_impute:
            if self.impute_outliers == "clip":
                _v[idxs_to_impute] = np.clip(
                    _v[idxs_to_impute], a_min=self.bounds[0], a_max=self.bounds[1]
                )
            elif self.impute_method == "mean":
                _v[idxs_to_impute] = np.nanmean(_v)
            elif self.impute_method == "median":
                _v[idxs_to_impute] = np.nanmedian(_v)

        _tmin, _tmax = self.encoding_range

        _v = ((_v - self.fitted_min) / (self.fitted_max - self.fitted_min) * (_tmax - _tmin)) + _tmin

        if self.obs_min_max is None:
            self.obs_min_max = (np.nanmin(_v), np.nanmax(_v))

        return {f"{self.target}": _v}

    def fit_encode(self, values):
        success = self.fit(values)
        # log.info(f"{success} in fitting {self.target}")
        return self.encode(values)

    def inverse(self, values):
        _tmin, _tmax = self.encoding_range

        return ((values - _tmin) * (self.fitted_max - self.fitted_min) / (_tmax - _tmin)) + self.fitted_min

    @property
    def metadata(self):
        return {
            str(self.target): {
                "output_dimension": 1,
                "output_type": "float",
                "num_variables": 1,
                "enc_type": "numeric",
                "variable_names": [f"{self.target}"],
            }
        }


class NuemricStandardizeProcessor(Processor):
    _structural_state_keys = Processor._structural_state_keys + (
        "mean", "standard_deviation", "seen_samples", "bounds",
    )

    def __init__(
        self,
        target: str,
        impute_method: str = "clip",
        impute_outliers: bool = False,
    ):
        """
        Standardize all continuous values to mean == zero, stdev ==1
        Parameters
        ----------
        target : the column name of the target variable
        impute_method : string - either "clip" or "mean" or "median"
        impute_outliers : boolean - if we want to impute outliers to more usable values
        """
        super().__init__(target)
        self.eps = 1e-12

        self.impute_method = impute_method.lower()
        self.impute_outliers = impute_outliers

        self.mean = 1
        self.standard_deviation = 1
        # track the number of samples
        self.seen_samples = 0
        self.bounds = ()

    def fit(self, values):
        if isinstance(values, dict) or isinstance(values, pd.DataFrame):
            _v = values.get(self.target).dropna(axis=0)
        else:
            _v = values

        if self.impute_outliers:
            self.bounds = calcualte_iqr_bounds(values, extension=2.0)

        self.mean = float(np.nanmean(values))
        self.standard_deviation = float(np.nanstd(values))
        self.seen_samples += len(values)
        self._fitted = True
        return True

    def encode(self, values):
        if isinstance(values, dict) or isinstance(values, pd.DataFrame):
            _v = values.get(self.target).dropna(axis=0)
        else:
            _v = values
        # --- outlier removal ---
        idxs_to_impute = set()

        if self.impute_outliers:
            idxs_to_impute, self.bounds = find_outliers(
                values=_v, idxs_to_impute=idxs_to_impute, bounds=self.bounds
            )

        if idxs_to_impute:
            if self.impute_outliers == "clip":
                _v[idxs_to_impute] = np.clip(
                    _v[idxs_to_impute], a_min=self.bounds[0], a_max=self.bounds[1]
                )
            elif self.impute_method == "mean":
                _v[idxs_to_impute] = np.nanmean(_v)
            elif self.impute_method == "median":
                _v[idxs_to_impute] = np.nanmedian(_v)

        _v = (_v - self.mean) / (self.standard_deviation + self.eps)

        if self.obs_min_max is None:
            self.obs_min_max = (np.nanmin(_v), np.nanmax(_v))

        return {f"{self.target}": _v}

    def fit_encode(self, values):
        success = self.fit(values)
        # log.info(f"{success} in fitting {self.target}")
        return self.encode(values)

    def inverse(self, values):
        return (self.standard_deviation - self.eps) * values + self.mean

    @property
    def metadata(self):
        return {
            str(self.target): {
                "num_variables": 1,
                "output_dimension": 1,
                "enc_type": "numeric",
                "output_type": "float",
                "variable_names": [f"{self.target}"],
            }
        }

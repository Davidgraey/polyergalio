"""
------------------------ Categorical preprocessing / encoders ------------------------------
--------------------------------------------------------------------------------------------
"""

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from polyergalio.encoders.encoders import Processor


class CategoricalProcessor(Processor):
    state_names = ("categorical_mapping", "value_mapping", "variable_counts", "rare_labels", "rare_value")

    def __init__(
        self,
        target: str,
        rare_encoding_threshold: Optional[float] = None,
        min_classes_for_rare_qualification: int = 15,
        for_target: bool = False,
    ):
        """
        Processor class - for encoding categorical variables.
        This processor will encode values into representation integer, but NOT into one-hot.
        This is by design - it's intended to be used in conjunction with a "embedding" layer, such as torch.embedding -
        that  serves as a lookup for categorical values and conversion into a one-hot -ish vector
        Parameters
        ----------
        target : The column name of the intended variable
        rare_encoding_threshold : float between 0.0 and 1.0 - for a 'rare' variable, it must have a frequence < this
            rare encoding threshold
        min_classes_for_rare_qualification : Before we start considering the placeholder 'rare label' - we must have
            at least min_classes already represented - so if we set 10, we would only consider using the 'rare label'
            placeholder if we have > 10 total variables
        for_target: bool - True to use target encoding
        """
        super().__init__(target)
        self.vari_name = f"{target}"
        self.categorical_mapping: dict = {}
        self.value_mapping: dict = {}
        self.variable_counts: Counter = Counter()
        self.for_target = for_target

        self.rare_encoding_threshold: float = rare_encoding_threshold
        self.min_classes_for_rare_qualification: int = min_classes_for_rare_qualification

        self.rare_key: str = "rare_label"
        self.rare_value: int = 999
        self.rare_labels: set = set()

    def count_categories(self, values: pd.Series) -> dict:
        """
        Get frequency (value_counts) for all
        Parameters
        ----------
        values : vector of our intended variable's values - flexible dtype, but we will create a class per unique
            value until the rare_qualify is reached

        Returns
        -------
        per-value frequency (held in self.variable_counts)
        """
        if isinstance(values, pd.Series):
            _counts = values.value_counts(dropna=True).to_dict()
            _counts = {k: int(v) for k, v in _counts.items()}

        if isinstance(values, pd.DataFrame):
            # logging for keyerror needed
            _counts = values[self.target].value_counts().to_dict()
            _counts = {k: int(v) for k, v in _counts.items()}

        return {
            k: v for k, v in sorted(_counts.items(), key=lambda kv: kv[1], reverse=True)
        }

    def fit(self, values: iter) -> bool:
        """
        Fit this encoder's components on the vector of data
        Parameters
        ----------
        values : vector of our intended variable's data

        Returns
        -------
        Bool - True if we were successful in fitting
        """
        values = values.dropna(axis=0)

        self.variable_counts = self.count_categories(values)

        items = list(self.variable_counts.keys())
        if self.min_classes_for_rare_qualification:
            if len(items) <= self.min_classes_for_rare_qualification:
                valids = items
                rare_labels = []
            else:
                valids = items[: self.min_classes_for_rare_qualification]
                rare_labels = items[self.min_classes_for_rare_qualification :]
        else:
            raise NotImplementedError("not yet implemented")

        self.rare_labels = rare_labels

        if self.for_target is True:
            enum_idx = 0
        else:
            enum_idx = 1
        self.categorical_mapping = {
            vari: int(cat_idx + enum_idx) for cat_idx, vari in enumerate(valids)
        }
        # since variables are enumerate(), we'll need to add to create the 'out of range' rare value
        self.rare_value = int(len(valids) + enum_idx)
        self.fitted = True

        # flip for lookup --
        self.value_mapping = {v: k for k, v in self.categorical_mapping.items()}

        return True

    def encode(self, values: pd.Series | pd.DataFrame) -> dict[str, iter]:
        """
        transform the values into their encoded representation
        Parameters
        ----------
        values : the intended variable's values - be sure that these match the data used to fit the processor

        Returns
        -------
        returns a dict of {"variable_name": [encoded_values]}
        """
        if self.fitted is False:
            raise ValueError("Not yet fitted")

        # modified - assuming Series is passed in & removing the array overheads
        results = []
        for one_value in values:
            if pd.isna(one_value):
                results.append(None)
            else:
                results.append(self.categorical_mapping.get(one_value, self.rare_value))

        if self.obs_min_max is None:
            self.obs_min_max = (0, max(list(self.value_mapping.keys())))
        return {f"{self.target}": np.array(results)}

    def fit_encode(self, values: iter) -> iter:
        """
        use the processor's fit, then encode processes in sequence
        Parameters
        ----------
        values : intended variable's values

        Returns
        -------
        returns the results of the encode process
        """
        success = self.fit(values)
        assert success is True
        _v = self.encode(values)
        return _v

    def inverse(self, values: iter) -> iter:
        """
        Invert the encoded vector back to the original values - this is NOT fully eq if there are any rare
            labels
        Parameters
        ----------
        values : encoded values (output from a previous self.encode() process)

        Returns
        -------
        vector of reverse-transformed values
        """
        if self.fitted is False:
            raise ValueError("Not yet fitted")

        # flip self.categorical_mapping
        # modified - assuming Series is passed in & removing the array overheads
        results = []
        for one_value in values:
            if pd.isna(one_value):
                results.append(None)
            else:
                results.append(self.value_mapping.get(one_value, self.rare_key))

        return {f"{self.target}": np.array(results)}

    @property
    def metadata(self):
        return {
            str(self.target): {
                "num_categories": len(self.categorical_mapping) + 1,
                "output_dimension": 1,
                "output_type": "int",
                "enc_type": "categorical",
                "rare_labels": self.rare_labels,
                "counts": self.variable_counts,
                "variable_names": [f"{self.target}"],
            }
        }

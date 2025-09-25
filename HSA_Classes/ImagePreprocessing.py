# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as PCA
from sklearn.preprocessing import StandardScaler
from loguru import logger
from loguru_config import LoguruConfig
import os

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")


class HSA_Image_preprocessing:
    def __init__(
        self,
        decomposer: PCA,
        scaler: StandardScaler,
        raw_path: str = "",
    ):
        self.raw_path = raw_path
        self.preprocessed_df = pd.read_pickle(self.raw_path)
        self.decomposer = decomposer
        self.scaler = scaler

    def read_raw_get_dummies(
        self,
        drop_keys: list = [],
        max_spawn_dummies: int = 0,
    ):
        """Read dataframe for single user and get dummies for keys containing categorical data and objects
        if max_spawn_list dummies is given will drop keys that generate more dummies than specified.
        the data frame is then scaled in preparation of pca."""

        df = pd.read_pickle(self.raw_path)
        df.drop(drop_keys, axis=1, inplace=True)
        for k in df.keys():
            if df[k].dtype in ["category", "object"]:
                if len(df[k].unique()) > max_spawn_dummies:
                    df.drop(k, axis=1, inplace=True)
                else:
                    d = pd.get_dummies(df[k])
                    for dk in d.keys():
                        df[dk] = d[dk]
                    df.drop(k, axis=1, inplace=True)
        scaled = self.scaler.fit_transform(df)
        df = pd.DataFrame(scaled, columns=df.keys())
        self.df = df

    def select_number_comps(
        self,
        percent_variance_explained: float = 0.95,
        min_additional_percent_variance_exp: float = 0.01,
    ):
        """Pass the decomposer of choice, pca, and both the percent_variance to explain,
        and the minimum percent of the variance that the addition of another component
        must achieve. Loops will break when percent_variance_explained is achieved, or when
        min_additional_percent_variance_exp is not achieved."""

        pca = self.decomposer.fit(self.df)
        additional_percent_variance = []
        sum_exp_var = 0
        for number_components in range(len(pca.explained_variance_ratio_)):
            temp = sum_exp_var
            sum_exp_var += pca.explained_variance_ratio_[number_components]
            additional_percent_variance.append(sum_exp_var - temp)
            if sum_exp_var > percent_variance_explained:
                print(
                    f"{number_components} components account for %{np.round(100*sum_exp_var,2)} of variance\nAcheived %{100*percent_variance_explained}"
                )
                break
            if additional_percent_variance[-1] < min_additional_percent_variance_exp:
                print(
                    f"{number_components} components account for %{np.round(100*sum_exp_var,2)} of variance\nMore features add less than %{100*min_additional_percent_variance_exp} explanation of variance"
                )
                break
        self.decomposer.set_params(n_components=number_components)
        self.df = self.decomposer.fit_transform(self.df)
        self.number_components = number_components

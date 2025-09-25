# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger
from loguru_config import LoguruConfig
import os

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")


class HSA_preprocessing:

    def __init__(
        self,
        decomposer: PCA,
        scaler: StandardScaler,
        logger: logger,
        raw_path: str = "",
        df: str = "",
        max_spawn_dummies: int = 500,
        allow_dict: dict = None,
    ):
        self.logger = logger
        self.max_spawn_dummies = max_spawn_dummies
        self.allow_dict = allow_dict

        def allow_lister(df, allow_dict):

            def str_to_list(
                row,
                allow_dict,
            ):
                for allow_type in allow_dict:
                    for key in allow_dict[allow_type].keys():
                        row[key] = row[key].split()
                return row

            def keep_if_one(row, allow_dict):

                for df_keys in allow_dict["keep_if_one"].keys():
                    cond = 0

                    df_set = set(row[df_keys])
                    dict_set = set(allow_dict["keep_if_one"][df_keys])
                    if not df_set.isdisjoint(dict_set):
                        cond += 1

                if cond > 0:
                    return row

                else:
                    row[df_keys] = (
                        np.nan
                    )  # cannot figure out how to drop here.. make nan and dropna
                    return row

            def drop_if_one(row, allow_dict):
                cond = 0

                for df_keys in allow_dict["drop_if_one"].keys():
                    df_set = set(row[df_keys])
                    dict_set = set(allow_dict["drop_if_one"][df_keys])

                    if df_set.isdisjoint(dict_set):
                        cond += 1

                if cond > 0:
                    return row

                else:
                    row[df_keys] = (
                        np.nan
                    )  # cannot figure out how to drop here.. make nan and dropna
                    return row

            def drop_if_all(row, allow_dict):
                cond = 0

                for df_keys in allow_dict["drop_if_all"].keys():
                    df_set = set(row[df_keys])
                    dict_set = set(allow_dict["drop_if_all"][df_keys])

                    if df_set.issubset(dict_set):
                        cond += 1

                if cond == len(allow_dict["drop_if_all"].keys()):
                    row[df_keys] = (
                        np.nan
                    )  # cannot figure out how to drop here.. make nan and dropna
                    return row

                else:
                    return row

            def keep_if_all(row, allow_dict):
                cond = 0

                for df_keys in allow_dict["keep_if_all"].keys():
                    df_set = set(row[df_keys])
                    dict_set = set(allow_dict["keep_if_all"][df_keys])

                    if df_set.issubset(dict_set):
                        cond += 1

                if cond == len(allow_dict["keep_if_all"].keys()):
                    return row

                else:
                    row[df_keys] = (
                        np.nan
                    )  # cannot figure out how to drop here.. make nan and dropna
                    return row

            args = (allow_dict,)
            df = df.apply(
                str_to_list,
                args=args,
                axis=1,
            )

            if len(self.allow_dict["keep_if_one"]):
                df = df.apply(
                    keep_if_one,
                    args=args,
                    axis=1,
                )

            if len(self.allow_dict["drop_if_one"]):
                df = df.apply(
                    drop_if_one,
                    args=args,
                    axis=1,
                )

            if len(self.allow_dict["drop_if_all"]):
                df = df.apply(
                    drop_if_all,
                    args=args,
                    axis=1,
                )

            if len(self.allow_dict["keep_if_all"]):
                df = df.apply(
                    keep_if_all,
                    args=args,
                    axis=1,
                )

            df.dropna(inplace=True, axis=0)
            return df

        def multi_connection_data_encoder(
            key: str,
            df: pd.DataFrame,
            max_spawn_dummies: int,
        ):
            """Takes a DF with space separated data. The data point with multiple connections are encoded to columns of the data frame. The new encoded columns are populated with the percent the value represents in the event.

            - key : key of the multivalued data
            - df : DF containing data
            - max_spawn_dummies : limit to the number of columns encoder can spawn
            """

            def str_to_list_mc(
                row,
                key,
                row_set,
            ):
                if type(row[key]) == list:
                    row[f"{key}_list"] = row[key]
                elif type(row[key]) == str:
                    row[f"{key}_list"] = row[key].split()
                return row

            def list_to_set(
                row,
                key,
                row_set,
            ):
                if type(row[f"{key}_list"]) == list:
                    row_set += list(set(row[f"{key}_list"]))
                else:
                    row_set += [row[f"{key}_list"]]
                row_set = set(row_set)
                try:
                    row_set.remove(np.nan)
                except Exception as e:
                    self.logger.critical(f"list_to_set: {e}")
                    sys.exit()
                return row_set

            def set_to_encoded_df_cols(
                row,
                key,
                row_set,
            ):
                if type(row[f"{key}_list"]) == list:
                    for spawn_key in row[f"{key}_list"]:
                        row[f"{spawn_key}_{key}"] += 1 / (len(row[f"{key}_list"]))
                else:
                    row[f'{row[f"{key}_list"]}_{key}'] = 1
                return row

            def clean_up(
                df,
                key,
            ):
                df.drop(key, inplace=True, axis=1)
                df.drop(f"{key}_list", inplace=True, axis=1)
                return df

            row_set = []
            args = (key, row_set)
            df = df.apply(
                str_to_list_mc,
                args=args,
                axis=1,
            )

            list_of_lists = list(df[f"{key}_list"])
            row_set = set()
            for sublist in list_of_lists:
                row_list = []
                if type(sublist) == list:
                    for element in sublist:
                        row_list.append(element)
                else:
                    row_list.append(sublist)
                row_set = row_set.union(set(row_list))

            if len(row_set) < max_spawn_dummies:
                for spawn_key in row_set:
                    temp = pd.DataFrame(
                        data=np.zeros(len(df)),
                        columns=[f"{spawn_key}_{key}"],
                    )

                    df = pd.concat(
                        [df, temp],
                        axis=1,
                    )

                df = df.apply(
                    set_to_encoded_df_cols,
                    args=args,
                    axis=1,
                )

            df = clean_up(
                df,
                key,
            )
            return df

        def allow_list(
            df: pd.DataFrame,
        ):
            """Casts columns of SPLUNK data to predefined datatype, else drops drill-down information from model consideration."""
            if self.allow_dict:
                df = allow_lister(df, self.allow_dict)
                df.dropna(inplace=True)
            return df

        if len(raw_path) > 0:
            self.raw_path = raw_path
            df = allow_list(pd.read_pickle(self.raw_path))
        else:
            df = allow_list(df)

        self.preprocessed_df = df

        self.decomposer = decomposer
        self.scaler = scaler
        self.logger.debug("Scaler and Decomposer passed to df_dtype gen.")

    def read_raw_get_dummies(
        self,
        max_spawn_dummies: int = 0,
    ):
        """Read dataframe for single user and get dummies for keys containing categorical data and objects. If max_spawn_list dummies is given will drop keys that generate more dummies than specified. The data frame is then scaled in preparation of pca."""

        df = self.preprocessed_df
        max_spawn_list = []

        for k in df.keys():
            if df[k].dtype in ["category", "object"]:
                if len(df[k].unique()) > max_spawn_dummies:
                    df.drop(k, axis=1, inplace=True)
                    max_spawn_list.append(k)

        d = df[df.select_dtypes(include=["category", "object"]).columns]
        if len(d.keys()):
            dummies = pd.get_dummies(d)
            df.drop(d_keys, axis=1, inplace=True)
            df = pd.concat([df, dummies], axis=1)

        time_list = []
        if "duration" in df.keys():
            if df["duration"].dtype != float:
                for t in df["duration"]:
                    time_list.append(t.total_seconds())
                df.drop("duration", axis=1, inplace=True)
                df["duration"] = time_list

        df.columns = df.columns.astype(str)

        for k in df.keys():
            if df[k].nunique() == 1:
                df.drop(k, inplace=True, axis=1)
                self.logger.info(f"Key: {k} Dropped due to STD == 0")
        scaled = self.scaler.fit_transform(df)

        df = pd.DataFrame(scaled, columns=df.keys())
        self.preprocessed_df = df
        self.logger.trace("Preprocessing Encoding Complete. ")
        for key in self.preprocessed_df.keys():
            if df[key].isnull().any():
                if len(self.preprocessed_df[key].unique()) == 1:
                    self.preprocessed_df.drop(key, axis=1, inplace=True)
        self.preprocessed_df.dropna(inplace=True)

        return max_spawn_list

    def select_number_comps(
        self,
        percent_variance_explained: float = 0.95,
        min_additional_percent_variance_exp: float = 0.01,
        df: pd.DataFrame = None,
    ):
        """Pass the decomposer of choice, pca, and both the percent_variance to explain,
        and the minimum percent of the variance that the addition of another component
        must achieve. Loops will break when percent_variance_explained is achieved, or when
        min_additional_percent_variance_exp is not achieved."""

        print(df)
        pca = self.decomposer.fit(df)
        differences_list = []
        sum_exp_var = 0
        per_exp = percent_variance_explained
        min_additional_percent_variance_exp = min_additional_percent_variance_exp
        for n_components in range(len(pca.explained_variance_ratio_)):
            temp = sum_exp_var
            sum_exp_var += self.decomposer.explained_variance_ratio_[n_components]
            differences_list.append(sum_exp_var - temp)
            select_comps = ""
            if sum_exp_var > per_exp:
                select_comps = f"{n_components} components account for %{np.round(100*sum_exp_var,2)} of variance\nAchieved %{100*percent_variance_explained}"
                break
            if differences_list[-1] < min_additional_percent_variance_exp:
                select_comps = f"{n_components} components account for %{np.round(100*sum_exp_var,2)} of variance\nMore features add less than %{100*min_additional_percent_variance_exp} explanation of variance"
                break
        self.decomposer.set_params(n_components=n_components)
        self.np = self.decomposer.fit_transform(df)
        self.n_components = n_components
        self.logger.debug(
            "Number of components selected by percent variance explained completed."
        )
        return select_comps

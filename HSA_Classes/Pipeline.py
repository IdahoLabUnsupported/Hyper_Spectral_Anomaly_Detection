# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
from copy import deepcopy
import sys
import os
import datetime
import Preprocessing as hsa_preprocessing
import DataSet as hsa_dataset
import Model as hsa_model
import utils
from sklearn.decomposition import PCA as PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader

from loguru import logger
from loguru_config import LoguruConfig

import os

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")


class HSA_pipeline:

    def __init__(
        self,
        penalty_ratio: float,
        cutoff_distance: float,
        lr: float,
        anomaly_std_tolerance: float,
        bin_count: int,
        percent_variance_explained: float,
        min_additional_percent_variance_exp: int,
        verbose: bool = 0,
        batch_size: int = 2000,  # Batch size for data loader
        multi_filters: int = 15,  # How many multifilter steps to take
        converge_toll: float = 1e-30,  # Defines convergence of anomaly score during opt
        affinity_matrix_iterations: int = 20,  # Number of powers of affinity matrix to generate
        iterations: int = int(5e3),  # 5e2 Number of optim.adam steps
        base_directory: str = None,
        unique_id_str: str = "",
        num_workers: int = 10,
        raw_path: str = None,
        max_spawn_dummies: int = 100,
        allow_dict: dict = None,
    ):
        self.penalty_ratio = penalty_ratio
        self.cutoff_distance = cutoff_distance
        self.lr = lr
        self.anomaly_std_tolerance = anomaly_std_tolerance
        self.bin_count = bin_count
        self.max_spawn_dummies = max_spawn_dummies
        self.percent_variance_explained = percent_variance_explained
        self.min_additional_percent_variance_exp = min_additional_percent_variance_exp
        self.verbose = verbose
        self.logger = logger
        self.unique_id_str = unique_id_str
        self.raw_path = raw_path
        self.allow_dict = allow_dict
        self.logger.trace("HSA_pipeline has been generated")
        self.logger.info(
            f"penalty_ratio: {self.penalty_ratio}, cutoff_distance: {self.cutoff_distance}, lr: {self.lr}, anomaly_std_tolerance: {self.anomaly_std_tolerance}, bin_count: {self.bin_count}, max_spawn_dummies: {self.max_spawn_dummies}, percent_variance_explained: {self.percent_variance_explained}"
        )

        self.base_directory = base_directory
        self.plot_directory = f"{self.base_directory}plots"  # Storage location
        self.log_directory = f"{self.base_directory}logs"  # Storage location
        self.results_directory = f"{self.base_directory}results"  # Storage location

        self.batch_size = batch_size
        self.multi_filters = multi_filters
        self.converge_toll = converge_toll
        self.affinity_matrix_iterations = affinity_matrix_iterations
        self.iterations = iterations
        self.num_workers = num_workers

        dir_list = [
            self.base_directory,
            self.plot_directory,
            self.log_directory,
            self.results_directory,
        ]
        try:
            for directory in dir_list:
                os.makedirs(directory, exist_ok=True)
            self.logger.debug("All output directories currently exist.")
        except Exception as e:
            self.logger.critical(f"An output directory is missing.\n {e}")
            sys.exit()

    def pipeline(
        self,
        df: pd.DataFrame = None,
        if_preprocess: bool = True,
    ):

        self.logger.debug("HSA_pipeline.pipeline() has begun")
        time_start = time.time()
        df_raw = deepcopy(df)

        pca = PCA()
        scaler = StandardScaler()

        prep = hsa_preprocessing.HSA_preprocessing(
            decomposer=pca,
            scaler=scaler,
            logger=self.logger,
            df=df,
            raw_path=self.raw_path,
            max_spawn_dummies=self.max_spawn_dummies,
            allow_dict=self.allow_dict,
        )

        if if_preprocess:
            self.logger.info("Preprocessing has begun.")
            max_spawn_list = prep.read_raw_get_dummies(
                max_spawn_dummies=self.max_spawn_dummies
            )

            select_comps = prep.select_number_comps(
                percent_variance_explained=self.percent_variance_explained,
                min_additional_percent_variance_exp=self.min_additional_percent_variance_exp,
                df=prep.preprocessed_df,
            )

            self.logger.debug(
                "Preprocessing is complete, data has been processed as a np.array ready for use in Torch."
            )
            if len(prep.np) == 0:
                preprocess_warning = (
                    "Did not return data from preprocessing! HSA_Preprocessing.py"
                )
                self.logger.critical(preprocess_warning)
                if self.verbose:
                    print(preprocess_warning)
                sys.exit()
        else:
            prep.np = prep.preprocessed_df.to_numpy()

        self.logger.trace("Hyper and Batch parameters are being passed to HSA_model.")

        model = hsa_model.HSA_model(
            penalty_ratio=self.penalty_ratio,
            cutoff_distance=self.cutoff_distance,
            converge_toll=self.converge_toll,
            anomaly_std_tolerance=self.anomaly_std_tolerance,
            affinity_matrix_iterations=self.affinity_matrix_iterations,
            lr=self.lr,
            init_anomaly_score=True,
            multifilter_flag=False,
        )
        self.logger.trace("Data is passed to HSA_dataset to make torch dataset.")
        dataset = hsa_dataset.HSA_dataset(prep.np)
        self.logger.info("Dataset is passed to Dataloader.")
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

        # Set storage location for outputs
        total_anomaly_index_list = (
            []
        )  # Unknown number of anomalies, using the append and convert method

        # Set logging and directories
        utils.set_directories(
            log_directory=self.log_directory, results_directory=self.results_directory
        )
        self.logger.success(
            f"Query Generated {len(df_raw)} samples. After PreProcessing step {len(prep.np)} were passed to the model."
        )
        self.logger.info(
            f"After {pca}, data shape: {prep.np.shape}, broken into {len(loader)} data loaders"
        )
        if if_preprocess:
            self.logger.info(
                f"Features dropped due to max_spawn_list: {max_spawn_list}"
            )

        try:
            self.logger.info("Starting to run through the dataloader on initial pass.")
            for i, data in enumerate(loader):  # setting up gpus
                model.set_trial(
                    start_idx=i * self.batch_size,
                    batch_size=self.batch_size,
                    unique_id_str=self.unique_id_str,
                )
                if (
                    self.verbose
                    and int(len(loader) / 8)
                    and i % int(len(loader) / 8) == 0
                ):
                    print(f"\rEpoch {np.round(100*i/len(loader),2)}%")

                # Model set up and weight generation
                model.read_data(
                    data_multifilter_df=data.squeeze(0)
                ).vertex_weights_distance().weight_generation().graph_evolution()
                # Training steps
                model.train(iterations=self.iterations)
                # Prediction step
                model.infer(
                    df=prep.preprocessed_df,
                    multifilter_flag=False,
                )
                # Store anomalous predictions throughout all batches for use in multi filter
                total_anomaly_index_list.append(model.anomaly_index_raw)
        except Exception as e:
            self.logger.critical(f"Initial pass FAILED. {e}")
            sys.exit()

        total_anomaly_index = np.concatenate(total_anomaly_index_list, axis=0)
        # Enstantuate np.ones a single time (Profileing Kameron)
        _ones = np.ones(len(prep.np))

        self.logger.success("Completed dataloader on initial pass.")
        mix_index, mix_data, anomaly_index = model.global_collect_multifilter_df(
            total_anomaly_index=prep.np,
            total_anomaly_index=total_anomaly_index[: len(prep.np)].astype(int),
            mf_batch_size=9 * len(total_anomaly_index),
            _ones=_ones,
        )
        anomaly_prediction_frequency_df = pd.DataFrame()
        anomaly_prediction_frequency_df["User DF Index"] = anomaly_index
        anomaly_prediction_frequency_df.set_index("User DF Index")
        anomaly_prediction_frequency_df["Anomaly Bin Count"] = np.zeros(
            len(anomaly_index)
        )

        # Randomly shuffle anomalies from all batches in unison
        model.uni_shuffle_multifilter_df(
            mix_index=mix_index.astype(int),
            mix_data=mix_data.astype(int),
            predicted_anomaly=anomaly_index.astype(int),
        )
        mf_data = model.all_data
        self.logger.debug(
            "Anomalous data has been colleted into first multifilter dataset."
        )

        user_location = model.all_index_user
        tend = time.time()
        if len(anomaly_index) > 0:
            if self.verbose:
                print(f"Count of 1st rank anomalies: {len(anomaly_index)}")
            self.logger.info(
                f"HSA 1st pass detected {len(anomaly_index)} to be passed to the multifilter."
            )
            ################################################################################
            # Multifilter Model
            try:
                for i in range(self.multi_filters):
                    batch_dataset = hsa_dataset.HSA_dataset(
                        preprocessed_np=mf_data,
                    )
                    batch_loader = DataLoader(
                        batch_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                    )
                    for j, data in enumerate(batch_loader):
                        # # Set up multi filter model
                        MF_model = hsa_model.HSA_model(
                            penalty_ratio=self.penalty_ratio,
                            cutoff_distance=self.cutoff_distance,
                            converge_toll=self.converge_toll,
                            anomaly_std_tolerance=self.anomaly_std_tolerance,
                            affinity_matrix_iterations=self.affinity_matrix_iterations,
                            lr=self.lr,
                            init_anomaly_score=True,
                            multifilter_flag=True,
                        )
                        MF_utils.set_directories(
                            log_directory=self.log_directory,
                            results_directory=self.results_directory,
                        )
                        MF_model.set_trial(
                            start_idx=(j - 1) * len(data),
                            batch_size=len(data),
                            unique_id_str=self.unique_id_str,
                        )
                        MF_model.read_data(
                            data_multifilter_df=data.squeeze(0),
                        ).vertex_weights_distances().weight_generation().graph_evolution()

                        # # Train MF_MODEL
                        MF_model.train(
                            iterations=self.iterations,
                        )
                        MF_model.infer(
                            df=prep.preprocessed_df,
                            multifilter_flag=True,
                            all_index_user=user_location,
                        )
                    anomaly_prediction_frequency_df.loc[
                        anomaly_prediction_frequency_df["User DF Index"].isin(
                            MF_model.anomaly_index_raw
                        ),
                        "Anomaly Bin Count",
                    ] += 1

                self.logger.trace(
                    f"Multifilter {i} of {self.multi_filters} multifilters is complete."
                )
                # # Global multifilter
                mix_index, mix_data, anomaly_index = (
                    model.global_collect_multifilter_df(
                        total_preprocessed_np=prep.np,
                        total_anomaly_index=total_anomaly_index[: len(prep.np)].astype(
                            int
                        ),
                        mf_batch_size=9 * len(total_anomaly_index),
                        _ones=_ones,
                    )
                )
                MF_model.uni_shuffle_multifilter_df(
                    mix_index=mix_index,
                    mix_data=mix_data,
                    predicted_anomaly=anomaly_index,
                )
                mf_data = MF_model.all_data
                user_location = MF_model.all_index_user
            except Exception as e:
                self.logger.critical(f"Multifilter FAILED on filter.  {e}")
                sys.exit()
            self.logger.success("Multifilter Complete.")
            self.logger.trace("Bin_predictions Heatmap Compete.")

            # Count how many times an anomaly occurs in the multifilter --> log
            for u in anomaly_prediction_frequency_df["Anomaly Bin Count"].unique():
                c = len(
                    anomaly_prediction_frequency_df[
                        anomaly_prediction_frequency_df["Anomaly Bin Count"] == u
                    ]
                )
                self.logger.info(
                    f"{c} Anomalies were predicted {u} times in MultiFilter"
                )

            pd.set_option("display.max_columns", None)
            pd.options.mode.copy_on_write = True
            results_df = df_raw.iloc[
                anomaly_prediction_frequency_df[
                    anomaly_prediction_frequency_df["Anomaly Bin Count"]
                    > self.bin_count
                ]["User DF Index"]
            ]
            results_df["Anomaly Bin Count"] = list(
                anomaly_prediction_frequency_df[
                    anomaly_prediction_frequency_df["Anomaly Bin Count"]
                    > self.bin_count
                ]["Anomaly Bin Count"]
            )
            results_df["User DF Index"] = list(
                anomaly_prediction_frequency_df[
                    anomaly_prediction_frequency_df["Anomaly Bin Count"]
                    > self.bin_count
                ]["User DF Index"]
            )
            if self.verbose:
                print(
                    "Results DF:",
                    results_df.sort_values("Anomaly Bin Count", ascending=False),
                )
            self.logger.trace("Results DF has been generated.")
            self.logger.debug(
                f"Standard run time {np.round((tend-time_start)/60,2)} for loader size of {self.batch_size}"
            )
            bin_percentage = (
                100
                * len(
                    anomaly_prediction_frequency_df[
                        anomaly_prediction_frequency_df["Anomaly Bin Count"]
                        > self.bin_count
                    ]["User DF Index"]
                )
                / len(anomaly_prediction_frequency_df)
            )

            self.logger.debug(
                "Parameter df has been updated with multifilter consistency metrics"
            )
            if self.verbose:
                print("-------------------------\n")
                time2 = time.time()
                print(
                    f"Inference time on {len(df_raw)} samples took {np.round(time2-time_start, 2)} sec"
                )

            results_file = (
                f"Results_{datetime.datetime.today().strftime("%Y-%m-%d")}.json"
            )

            path = f"{self.results_directory}/"
            write = path + results_file

            if "level_0" in results_df.keys():
                results_df.drop("level_0", axis=1, inplace=True)

            results_df.set_index("User DF Index").to_json(write, orient="records")

            self.logger.success(
                f"Results DF saved to {write}, run is complete. There are {len(results_df)} anomalies predicted."
            )
            return results_df
        else:
            self.logger.success(
                f"No Anomalies found in data during 1st HSA pass. COMPLETE {datetime.datetime.today().strftime("%Y-%m-%d")}"
            )

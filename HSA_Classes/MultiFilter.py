# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import Model as hsa_model
import DataSet as hsa_dataset
from torch.utils.data import DataLoader
import numpy as np
import cProfile

from loguru import logger
from loguru_config import LoguruConfig
import os

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")


def multifilter(
    multi_filters: int = 15,
    batch_size: int = 2000,
    penalty_ratio: float = 0.9,
    cutoff_distance: float = 2.5,
    anomaly_std_tolerance: float = 1,
    mf_data: np.ndarray = None,
    all_index_user: np.ndarray = None,
    pass1_model: hsa_model = None,
    update_console_ntimes: int = 1,
):

    profiler = cProfile.Profile()
    profiler.enable()
    _ones = np.ones(len(pass1_model.preprocessed_df))

    if multi_filters < update_console_ntimes:
        logger.warning(
            f"multi_filters >= update_console_ntimes: {multi_filters >= update_console_ntimes}, Resetting update_console_ntimes = multi_filters"
        )
        update_console_ntimes = multi_filters

    for j, mf_n in enumerate(range(multi_filters)):
        if j % int(multi_filters / update_console_ntimes) == 0:
            logger.success(f"Starting Multi Filter {j} of {multi_filters}")

        batch_dataset = hsa_dataset.HSA_dataset(mf_data)
        batch_loader = DataLoader(
            batch_dataset,
            batch_size=batch_size,
        )

        for data in batch_loader:
            # # Set up multi filter model
            MF_model = hsa_model.HSA_model(
                penalty_ratio=penalty_ratio,
                cutoff_distance=cutoff_distance,
                converge_toll=1e-5,
                anomaly_std_tolerance=anomaly_std_tolerance,
                affinity_matrix_iterations=20,
                lr=2.7,
                multifilter_flag=1,
            )
            MF_model.log_directory = pass1_model.log_directory
            MF_model.results_directory = pass1_model.results_directory
            MF_model.set_trial(j * len(data), len(data), pass1_model.unique_id_str)
            MF_model.read_data(
                data_multifilter_df=data.squeeze(0)
            ).vertex_weights_distances().weight_generation().graph_evolution()

            # # Train MF_MODEL
            MF_model.train(iterations=pass1_model.iterations)
            MF_model.infer(
                pass1_model.preprocessed_df,
                multifilter_flag=1,
                all_index_user=all_index_user,
            )

            pass1_model.anomaly_prediction_frequency_df.loc[
                pass1_model.anomaly_prediction_frequency_df["User DF Index"].isin(
                    MF_model.anomaly_index_raw
                ),
                "Anomaly Bin Count",
            ] += 1

            logger.trace(
                f"Multifilter {mf_n} of {multi_filters} multifilters is complete."
            )
            # # Global multifilter
            mix_index, mix_data, anomaly_index = (
                pass1_model.global_collect_multifilter_df(
                    pass1_model.preprocessed_df.to_numpy(),
                    pass1_model.total_anomaly_index[
                        : len(pass1_model.preprocessed_df.to_numpy())
                    ].astype(int),
                    9 * len(pass1_model.total_anomaly_index),
                    _ones,
                )
            )
            MF_model.uni_shuffle_multifilter_df(mix_index, mix_data, anomaly_index)
            mf_data = MF_model.all_data
            all_index_user = MF_model.all_index_user
    profiler.disable()
    profiler.print_stats(sort="cumulative")

    return MF_model, pass1_model.anomaly_prediction_frequency_df

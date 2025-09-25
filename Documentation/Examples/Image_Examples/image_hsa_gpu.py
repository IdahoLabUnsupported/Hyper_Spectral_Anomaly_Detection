# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from PIL import Image
from torch.utils.data import DataLoader
from loguru import logger
from loguru_config import LoguruConfig
import time
import json
import cProfile  # kept for prfiling

# import pstats # kept for prfiling

# Specify trial directory and update temp loguru config before passing to logger.
path = "../../../HSA_Classes"
sys.path.append(path)
import utils

base_directory = "./image_gpu_outputs/"

multi_filters = 5
unique_id_str = f"globular_cluster-mf{multi_filters}"

# Apply the configuration to the logger and ensure direcotries exist.
utils.edit_loguru_config(path, base_directory, unique_id_str)
log_directory, results_directory = utils.set_directories(
    f"{base_directory}{unique_id_str}/logs/",
    results_directory=f"{base_directory}{unique_id_str}/results/",
)

start_time = time.time()
# -------------------------------------------------------
# -------------------------------------------------------
profiler = cProfile.Profile()
image_width = 753  #  618 #1280
anomaly_std_tolerance = 0.8
penalty_ratio = 0.099
cutoff_distance = 2
converge_toll = 1e-5

batch_size = 4000
iterations = 10000
lr = 1e-3

image_path = "./glob.jpeg"
update_console_ntimes = 20  # how many times to output to logs when itterating data in dataloader and filter in multifilter.

img = Image.open(image_path)
array = np.array(img)

logger.success(f"Input array shape: {array.shape}")

data = None
columns = ["red", "green", "blue"]
data = utils.reshape_a_b_c__ab_c(array)

scaler = MaxAbsScaler()
scaler.fit(data)
preprocessed_df = pd.DataFrame(scaler.transform(data), columns=columns)

import Model as hsa_model
import DataSet as hsa_dataset
import MultiFilter as hsa_multifilter

model = hsa_model.HSA_model(
    penalty_ratio=penalty_ratio,
    cutoff_distance=cutoff_distance,
    converge_toll=converge_toll,
    anomaly_std_tolerance=anomaly_std_tolerance,
    affinity_matrix_iterations=20,
    lr=lr,
    multifilter_flag=0,
)
model.log_directory, model.results_directory = log_directory, results_directory

logger.info(f"Initial HSA model has been instantiated.")

dataset = hsa_dataset.HSA_dataset(
    preprocessed_np=preprocessed_df.to_numpy(),
)
logger.info(f"HSA DataSet has been instantiated.")

dataloader = DataLoader(dataset, batch_size=batch_size)
logger.info(f"HSA DataLoader has been instantiated.")

model.preprocessed_df = preprocessed_df
total_anomaly_index_list = []
# Since the number of anomalies predicted is unknown to preallocate data,
# using list append and conversion method.
if update_console_ntimes >= len(dataloader):
    log_directory.warning(
        f"update_console_ntimes >= len(dataloader): {update_console_ntimes >= len(dataloader)}, Resetting update_console_ntimes = len(dataloader)."
    )
    update_console_ntimes = len(dataloader)

for i, data in enumerate(dataloader):
    if i % int(len(dataloader) / update_console_ntimes) == 0:
        logger.success(f"Starting dataloader on epoch. Data: {i} of {len(dataloader)}")

    model.set_trial(i * batch_size, batch_size, unique_id_str)

    # Model set up and weight generation
    model.read_data(
        data_multifilter_df=data.squeeze(0)
    ).vertex_weights_distances().weight_generation().graph_evolution()

    model.train(iterations=iterations)
    model.infer(preprocessed_df, multifilter_flag=0)
    total_anomaly_index_list.append(model.anomaly_index_raw)

total_anomaly_index = np.concatenate(total_anomaly_index_list)
_ones = np.ones(len(preprocessed_df))  # Make this vector only one time.

mix_index, mix_data, anomaly_index = model.global_collect_multifilter_df(
    preprocessed_df.to_numpy(),
    total_anomaly_index[: len(preprocessed_df.to_numpy())].astype(int),
    min(len(preprocessed_df), 9 * len(total_anomaly_index)),
    _ones,
)
logger.success(f"First Pass HSA complete. len(anomaly_index): {len(anomaly_index)}")
anomaly_prediction_frequency_df = model.apf_df_generation(anomaly_index)

model.uni_shuffle_multifilter_df(
    mix_index.astype(int), mix_data.astype(int), anomaly_index.astype(int)
)
mf_data = model.all_data
logger.debug("Anomalous data has been colleted into first multifilter dataset.")

all_index_user = model.all_index_user
update_console_ntimes = min(multi_filters, 20)

MF_model, anomaly_prediction_frequency_df = hsa_multifilter.multifilter(
    multi_filters,
    batch_size,
    penalty_ratio,
    cutoff_distance,
    anomaly_std_tolerance,
    mf_data,
    all_index_user,
    model,
    update_console_ntimes,
)

preprocessed_df["Anomaly Bin Count"] = np.zeros(len(preprocessed_df))
anomaly_prediction_frequency_df.set_index("User DF Index", inplace=True)

preprocessed_df.loc[anomaly_prediction_frequency_df.index, "Anomaly Bin Count"] = (
    anomaly_prediction_frequency_df["Anomaly Bin Count"]
)

anomaly_score_list = []
for i in range(int(len(preprocessed_df) / image_width)):
    anomaly_score_list.append(
        preprocessed_df["Anomaly Bin Count"][
            i * image_width : (1 + i) * image_width
        ].values
    )

anomaly_score_np = np.vstack(anomaly_score_list)
np.save(f"{results_directory}anomaly_score_{unique_id_str}", anomaly_score_np)

end_time = time.time()

logger.success(
    f"Run Time: {(end_time -start_time)/60} min, with batch_size: {batch_size}"
)
logger.success("Run Complete")

# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pandas as pd
import argparse
from loguru import logger
from loguru_config import LoguruConfig
import os

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s",
    "--site",
    help="The site name as seen in the results directory path. Ex: site6",
)
parser.add_argument(
    "-m",
    "--model",
    help="The model name to run as seen in the results directory path. Ex: cb_custom",
)
parser.add_argument(
    "-k",
    "--key",
    help="The static key that remains the same through out all trials",
)
parser.add_argument(
    "-t",
    "--today",
    help="The base date used through out all trials in +%Y-%m-%d",
)
parser.add_argument("-n", "--num_trials", help="The number of trials to compare Ex: 8")
args = parser.parse_args()

site = args.site
static_key = args.key
model = args.model
path = f"/{site}/{model}/"  # Trial comp will only be used in dev.  Fix this path to the dev results dir
num_trials = int(args.num_trials)
today = args.today


def get_index_list(
    path: str = "", num_trials: int = 4, static_key: str = "first_uid", today=None
) -> list:
    results_index_list = []
    for r in range(1, 1 + num_trials):
        index_list = []

        try:
            file = f"{path}Results_{today}{r}.json"
            data = pd.read_json(file)

            for i in range(len(data)):
                index_list.append(data.loc[i][static_key])

        except Exception as e:
            print(f"Did NOT FIND: {file}\n{e}")
            index_list.append("")
        results_index_list.append(index_list)

    return results_index_list


def similarity_matrix(results_index_list: list, num_trials: int = 4):  # df
    similarity_matrix = np.ones([num_trials, num_trials]) * 100
    keys_list = []
    for i in range(len(results_index_list)):
        keys_list.append(f"Trial_{i}")
        for j in range(i + 1, len(results_index_list)):
            common = len(
                set(results_index_list[i]).intersection(set(results_index_list[j]))
            )
            max_len = max(len(results_index_list[i]), len(results_index_list[j]))

            if max_len:
                similarity_matrix[i, j] = np.round(100 * common / max_len, 2)
                similarity_matrix[j, i] = np.round(100 * common / max_len, 2)

            else:  # both have 0 pred
                similarity_matrix[i, j] = 100
                similarity_matrix[j, i] = 100

    similarity_df = pd.DataFrame(similarity_matrix, index=keys_list, columns=keys_list)
    similarity_df["mean_percent_similarity"] = similarity_df.mean()
    similarity_df["std_percent_similarity"] = similarity_df.std()

    return similarity_df


results_index_list = get_index_list(path, num_trials, static_key, today)
similarity_df = similarity_matrix(results_index_list, num_trials)
print(similarity_df)
print(
    f"Mean(Mean(percent_similarity): {np.round(similarity_df['mean_percent_similarity'].mean(),2)}%, Mean(STD(percent_similarity): {np.round(similarity_df['std_percent_similarity'].mean(),2)}%"
)
print(f"Mean predictions {np.round(np.mean([len(x) for x in results_index_list]),2)}")

# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from loguru import logger
from loguru_config import LoguruConfig
import os

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")


class HSA_viz:

    def __init__(
        self,
        m: list,
        preprocessed_np: np.ndarray,
        batch_size: int,
        start_idx: int,
        verbose: bool,
        plot_fig: bool,
        save_fig: bool,
        plots_directory: str,
        unique_id_str: str,
        logger: logger,
    ):
        self.m = m
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.verbose = verbose
        self.figures = plot_fig
        self.save_fig = save_fig
        self.plots_directory = plots_directory
        self.preprocessed_np = preprocessed_np
        self.unique_id_str = unique_id_str
        self.logger = logger

    def heatmap_weights_matrix(
        self,
        title_list: list,
        data_list: list,
    ):
        """Generates  heat-maps for edge weights, vertex weights, gamma and d matrices, and raw data"""

        if self.figures:
            fig, axes = plt.subplots(3, 2, figsize=(20, 35))
            c = 0
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    sns.heatmap(data_list[c], ax=axes[i, j], cmap="viridis")
                    axes[i, j].set_title(title_list[c])
                    c += 1
            if self.save_fig:
                fig.savefig(
                    f"{self.plots_directory}/{self.unique_id_str}_model_weights.png"
                )
            self.logger.debug(
                f"heatmap_weights_matrix was generated. The save_fig={self.save_fig}."
            )

    def heatmap_bin_predictions(
        self,
        v_min: float,
        bin_outputs: list,
        x_ticks: list,
        x_label: list,
    ):
        """Generates heat maps for model predictions, bin scores, and raw data.  X ticks and
        corresponding x labels indicated predicted anomaly"""

        if self.figures:
            fig, axes = plt.subplots(1, 3, figsize=(25, 30))
            sns.heatmap([self.m], ax=axes[0], cbar=False, cmap="viridis")
            axes[0].set_title("Raw Anomaly Score")
            axes[0].set_xlabel("Pixel")
            axes[0].set_ylabel("Score")
            axes[0].set_yticks([])

            sns.lineplot([bin_outputs], ax=axes[1], legend=False)
            axes[1].set_xlabel("Pixel")
            axes[1].set_ylabel("STD From Mean")
            axes[1].set_title("Model Prediction in STD from Mean")
            axes[1].set_xlim(0, len(bin_outputs))
            axes[1].set_ylim(0, 1.025 * max(bin_outputs))

            sns.heatmap(
                self.preprocessed_np.transpose(1, 0),
                ax=axes[2],
                cbar=False,
                cmap="viridis",
            )
            axes[2].set_xlabel("Pixel")
            axes[2].set_ylabel("Features")
            axes[2].set_title("Pixel Vector")

            axes[0].set_xticks(x_ticks)
            axes[1].set_xticks(x_ticks)
            axes[2].set_xticks(x_ticks)
            axes[0].set_xticklabels(x_label)
            axes[1].set_xticklabels(x_label, rotation=90)
            axes[2].set_xticklabels(x_label)
            if self.save_fig:
                fig.savefig(
                    f"{self.plots_directory}/{v_min}std_pred_{datetime.today().strftime('%Y-%m-%d')}.png"
                )
            self.logger.debug(
                f"heatmap_bin_predictions was generated. The save_fig={self.save_fig}."
            )

    def heatmap_bin_predictions_vert(
        self,
        bin_outputs: list,
        x_ticks: list,
        x_label: list,
    ):
        """Generates heat maps for model predictions, bin scores, and raw data.  X ticks and
        corresponding x labels indicated predicted anomaly"""

        if self.figures:
            fig, axes = plt.subplots(1, 3, figsize=(25, 30))
            # print(len([np.transpose(self.m)]))
            sns.heatmap(
                (np.array(self.m, ndmin=2).transpose()),
                ax=axes[0],
                cbar=False,
                cmap="viridis",
            )
            axes[0].set_title("Raw Anomaly Score")
            axes[0].set_ylabel("Pixel")
            axes[0].set_xlabel("Score")

            axes[1].plot(np.flip(bin_outputs), range(len(bin_outputs)))
            axes[1].set_ylabel("Pixel")
            axes[1].set_xlabel("STD From Mean")
            axes[1].set_title("Model Prediction in STD from Mean")
            axes[1].set_xlim(0, 1.025 * max(bin_outputs))
            axes[1].set_ylim(0, len(bin_outputs))

            sns.heatmap(
                (np.array(self.preprocessed_np, ndmin=2)),
                ax=axes[2],
                cbar=False,
                cmap="viridis",
            )
            axes[2].set_ylabel("Pixel")
            axes[2].set_xlabel("Features")
            axes[2].set_title("Pixel Vector")

            axes[0].set_yticks(x_ticks)
            axes[1].set_yticks(len(bin_outputs) - (x_ticks))
            axes[2].set_yticks(x_ticks)
            axes[0].set_yticklabels(x_label)
            axes[1].set_yticklabels(x_label)
            axes[2].set_yticklabels(x_label)
            if self.save_fig:
                fig.savefig(
                    f"{self.plots_directory}/std_pred_{datetime.today().strftime('%Y-%m-%d')}.png"
                )
            self.logger.debug(
                f"heatmap_bin_predictions_vert was generated. The save_fig={self.save_fig}."
            )

# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pandas as pd
import torch as t
import os

from loguru import logger
import os
from loguru_config import LoguruConfig

path = os.path.dirname(os.path.realpath(__file__))
# Apply the configuration to the logger
LoguruConfig.load(f"{path}/loguru_config.json")


class HSA_model:

    def __init__(
        self,
        penalty_ratio: float,
        cutoff_distance: float,
        converge_toll: float,
        anomaly_std_tolerance: float,
        affinity_matrix_iterations: int,
        lr: float,
        init_anomaly_score: bool = True,
        multifilter_flag: bool = False,
    ):
        self.logger = logger
        assert isinstance(
            affinity_matrix_iterations, int
        ), f"Affinity matrix iterations must be an int, not {type(affinity_matrix_iterations)}"
        assert isinstance(
            init_anomaly_score, bool
        ), f"Affinity matrix iterations must be an bool, not {type(init_anomaly_score)}"
        assert isinstance(
            multifilter_flag, int
        ), f"Affinity matrix iterations must be an bool, not {type(multifilter_flag)}"

        @self.logger.catch
        def get_free_gpu():
            """Looks at allocated memory on all available devices and returns device with most available memory. DEPRECATED by any scheduler software."""

            allocated_mem = 1000  # set arbitrarily large
            free_device = "cuda:0"
            if t.cuda.is_available():
                for device in range(t.cuda.device_count()):
                    device_name = f"cuda:{device}"
                    device = t.device(device_name)
                    if device.type == "cuda":
                        mem = t.cuda.memory_allocated(0) / 1024**3
                        if mem < allocated_mem:
                            free_device = device_name
                            allocated_mem = mem
            else:
                free_device = "cpu"
                self.logger.warning("No GPU available, running on CPU.")
            return free_device

        self.device = get_free_gpu()
        self.penalty_ratio = penalty_ratio
        self.cutoff_distance = cutoff_distance
        self.stopping_toll = converge_toll
        self.std_toll = anomaly_std_tolerance
        self.affinity_matrix_iterations = affinity_matrix_iterations
        self.init_anomaly_score = init_anomaly_score
        self.multifilter_flag = multifilter_flag
        self.lr = lr

    def set_trial(
        self,
        start_idx: int,
        batch_size: int,
        unique_id_str: str,
    ):
        assert isinstance(
            start_idx, int
        ), f"start_index must be an int, not {type(start_idx)}"
        assert isinstance(
            batch_size, int
        ), f"batch_size must be an int, not {type(batch_size)}"

        self.start_idx = start_idx
        self.batch_size = batch_size
        self.unique_id_str = unique_id_str
        self.logger.trace("HSA Model trials set.")
        return self

    def read_data(
        self,
        data_multifilter_df=None,
    ):
        """Generates the pixel vector to detect anomalies in.  Can also generate an anomaly at a given idx. This anomaly will be the mean of each feature by a random number of anomaly scale."""

        if type(data_multifilter_df) == t.Tensor:
            self.preprocessed_np = data_multifilter_df
        else:
            self.preprocessed_np = t.from_numpy(data_multifilter_df)
        self.batch_size = min(self.batch_size, len(self.preprocessed_np))
        self.user_plt = (
            self.results_directory
            + self.unique_id_str
            + f"_start_idx-{str(self.start_idx)}_num_samples-{str(self.batch_size)}"
        )
        if self.init_anomaly_score:
            self.anomaly_score_old = (
                np.ones(len(self.preprocessed_np)) * 1000
            )  # make the initial m for difference greater than stopping_toll
            self.m = np.random.rand(
                self.batch_size
            )  # make the initial m for difference greater than stopping_toll
        self.logger.debug("Initial random anomaly index set.")
        return self

    def vertex_weights_distances(
        self,
    ):
        """Defines model weights dependant on pixel separation in function space
        Returns both the pairwise distances and summed differences for each pixel set"""

        p = self.preprocessed_np.to(self.device)
        pl0 = len(self.preprocessed_np)
        p_mat = p.unsqueeze(0)
        p_mat = p_mat.expand(pl0, pl0, -1)
        p_matT = p_mat.transpose(0, 1)
        difference = t.subtract(p_mat, p_matT)
        sq_diff = t.square(difference)
        sum_sqr = t.sum(sq_diff, -1)

        self.distances = t.sqrt(sum_sqr).requires_grad_(False)
        self.vertex_weights = (
            t.sum(t.exp(-t.square(t.div(self.distances, self.cutoff_distance))), 1)
            .unsqueeze(-1)
            .requires_grad_(False)
            .type(t.DoubleTensor)
            .to(self.device)
        )
        self.edgeWeights = (
            t.exp(-t.div(self.distances, self.cutoff_distance**2))
            .requires_grad_(False)
            .type(t.DoubleTensor)
            .to(self.device)
        )
        self.logger.debug("HSA Model Weight Generated.")
        return self

    def weight_generation(
        self,
    ):
        """Equation 2: Vertex weights * Identity to generate gamma matrix in eq 7
        Equation 6: S\tilde (sim matrix here) use edge weights and scale with d_matrix  --CHOOSING TO USE GRB OVER EUCLIDEAN DIST
                d-matrix  sum of rows of sim_matrix * identity
        Equation 7: gamma*sim*gamma  ~vertex*edge*vertex info

        Returns the 3 [nxn] matrices for Affinity Matrix= aff_matrix, Gamma Matrix= gam_matrix,
        and D matrix= d_matrix"""

        d_vec = t.pow(t.sum(self.edgeWeights, 0), (-1 / 2))
        d_matrix = t.diag(d_vec)
        gam_matrix = t.diag(
            t.reshape(self.vertex_weights, (self.vertex_weights.shape[0],))
        )

        temp = t.matmul(d_matrix, self.edgeWeights)
        sim_matrix = t.matmul(temp, d_matrix)

        temp = t.matmul(gam_matrix, sim_matrix)
        aff_matrix = t.matmul(temp, gam_matrix)

        self.aff_matrix = aff_matrix
        self.sim_matrix = sim_matrix
        self.gam_matrix = gam_matrix
        self.d_matrix = d_matrix
        self.logger.debug("HSA Graphs generated.")
        return self

    def graph_evolution(
        self,
    ):
        """Edge weight information is evolved by powers of the affinity  and D matrices. Vertex weights do not need to evolve. Each power of these is needed for optimization of the quadratic in Equation 10. Returns List of sets of matrix to power:  [[A^1,A^2, A^3...A^k],[D^1,D^2, D^3...D^k]]"""

        matrix_list = [self.aff_matrix, self.d_matrix]
        sets = []
        for m in matrix_list:
            m_set = [m]
            for i in range(1, self.affinity_matrix_iterations):
                power = t.matmul(m_set[-1], m_set[0])
                m_set.append(power)
            sets.append(m_set)
        self.sets = sets
        self.logger.debug("HSA Graph Theory complete.")
        return self

    @t.compile
    def torch_POF(
        m: t.Tensor,
        affinity_m: t.Tensor,
        vertex_weights: t.Tensor,
        d_matrix: t.Tensor,
        power: t.Tensor,
        penalty_ratio: float,
        device: str,
    ):
        """Equation 10: The quadratic objective fxn  = obj [nx1]
        Equation 11: Best interpretation of constraint eqn/ Penalty terms
        Equation 12: Function to be minimized by choice of anomaly scores
            - best interpretation: no definition of U, no summation preformed here
        r: penalty scaling that must approach 0 as nu -->k=iter_steps"""

        assert isinstance(m, t.Tensor), f"m must be an troch.Tensor, not {type(m)}"
        assert isinstance(
            affinity_m, t.Tensor
        ), f"affinity_m must be an troch.Tensor, not {type(affinity_m)}"
        assert isinstance(
            vertex_weights, t.Tensor
        ), f"vertex_weights must be an troch.Tensor, not {type(vertex_weights)}"
        assert isinstance(
            d_matrix, t.Tensor
        ), f"d_matrix must be an troch.Tensor, not {type(d_matrix)}"
        assert isinstance(
            power, t.Tensor
        ), f"power (Affinity matrix iterations) must be an troch.Tensor, not {type(power)}"

        obj = (1 / 2) * t.matmul(
            t.matmul(t.transpose(m.unsqueeze(1), 0, 1), affinity_m), m.unsqueeze(1)
        )
        c = t.matmul(
            t.matmul(t.transpose(vertex_weights, 0, 1), d_matrix), m.unsqueeze(1)
        )
        neg_constraint = t.le(m, 0)
        ones = t.ones(m.size()).to(device)
        ge1_constraint = t.gt(t.subtract(m, ones), 0)

        constraint = t.mul(t.logical_or(neg_constraint, ge1_constraint).double(), m)
        pen = t.mul(
            t.pow(penalty_ratio, power), t.pow(constraint, 4)
        )  # 4 here is a set scaling of penalty ratio (arb could be changed)

        phi = obj + c + t.sum(pen)
        return phi.to(device)

    def train(
        self,
        torch_POF=torch_POF,
        iterations: float = 1e4,
    ):
        """Using the evolved edge weight information and initialized anomaly scores minimize the penalized objective fxn. Optimize for each power of the evolution using the previous best anomaly score."""
        self.iterations = iterations
        anomaly_score_old = t.from_numpy(self.anomaly_score_old).to(self.device)
        vert_weight = self.vertex_weights
        pr = t.tensor(self.penalty_ratio).requires_grad_(True).to(self.device)
        anomaly_score = t.from_numpy(self.m).to(self.device)

        def mac_opt_loop(
            _sets: list,
            _stopping_toll: float,
            _anomaly_score: t.Tensor,
            _pr: float,
            _vert_weight: t.tensor,
            _m_old: t.Tensor,
        ):
            assert isinstance(
                _anomaly_score, t.Tensor
            ), f"_anomaly_score must be an troch.Tensor, not {type(_anomaly_score)}"
            assert isinstance(
                _vert_weight, t.Tensor
            ), f"_vert_weight must be an troch.Tensor, not {type(_vert_weight)}"
            assert isinstance(
                _m_old, t.Tensor
            ), f"_m_old must be an troch.Tensor, not {type(_m_old)}"
            _m_mid = _m_old

            for i in range(len(self.sets[0])):
                m = _anomaly_score.requires_grad_(False)
                affinity_m = _sets[0][i]
                d_matrix = _sets[1][i]

                power = t.tensor(i + 1).requires_grad_(False)
                params = [m]

                optimizer = t.optim.Adam(params, lr=self.lr)

                # @t.compile  Cannot compile any aspect of the optimizer because the parameter we are
                # minimizing gets updated and passed to the optimizer. This deallocation breaks the computation graph. Instead the penalized objective function is compiled alone.
                def cmp_opt_step():
                    optimizer.step()

                # @t.compile
                def cmp_opt_zero_grad():
                    optimizer.zero_grad()

                for j in range(iterations):
                    cmp_opt_zero_grad()
                    loss = (
                        torch_POF(
                            m,
                            affinity_m,
                            _vert_weight,
                            d_matrix,
                            power,
                            _pr,
                            self.device,
                        )
                        .requires_grad_(True)
                        .to(self.device)
                    )
                    loss.backward()
                    cmp_opt_step()

                    _anomaly_score = params[0]
                    if t.le(
                        t.sqrt(t.sum((t.pow(t.sub(_anomaly_score, _m_mid), 2)))),
                        _stopping_toll,
                    ):
                        break
                    _m_mid = _anomaly_score

                if t.le(
                    t.sqrt(t.sum((t.pow(t.sub(_anomaly_score, _m_old), 2)))),
                    _stopping_toll,
                ):
                    break
                _m_old = (
                    _anomaly_score  # Update the previous best guess at anomaly score
                )
            return _anomaly_score

        stopping_toll = (
            t.tensor(self.stopping_toll).requires_grad_(False).to(self.device)
        )

        anomaly_score = mac_opt_loop(
            self.sets,
            stopping_toll,
            anomaly_score,
            pr,
            vert_weight,
            anomaly_score_old,
        )

        self.m = anomaly_score.cpu().detach().numpy()
        self.logger.debug("HSA Torch optimization complete.")

    def infer(
        self,
        df: pd.DataFrame = None,
        multifilter_flag: bool = False,
        all_index_user: list = None,
    ):
        """Measures the distance from the mean in std for each minimized anomaly score.
        Returns the number of std from mean if greater than the std_anomaly_thresh
        Returns the raw data for anomalous pixels in the bin_df
        Returns the x_ticks for heat-maps (location in sub preprocessed_np)
        Returns the anomaly_index_raw for heat-maps (location in total_preprocessed_np and raw data)
        """
        assert isinstance(
            df, pd.DataFrame
        ), f"df must be a pd.DataFrame, not {type(df)}"
        assert isinstance(
            multifilter_flag, int
        ), f"multifilter_flag must be a pd.DataFrame, not {type(multifilter_flag)}"
        m_mean = np.mean(self.m)
        m_std = np.std(self.m)

        self._filtered_zscore = abs(self.m - m_mean)

        self._filtered_zscore[
            np.where(self._filtered_zscore / m_std < self.std_toll)
        ] = 0
        self._filtered_zscore[
            np.where(self._filtered_zscore / m_std > self.std_toll)
        ] = np.round(
            self._filtered_zscore[
                np.where(self._filtered_zscore / m_std > self.std_toll)
            ]
            / m_std,
            1,
        )

        self.anomalous_location = np.where(self._filtered_zscore > 0)[0]

        if multifilter_flag:
            self.anomaly_index_raw = all_index_user[self.anomalous_location]
        else:
            self.anomaly_index_raw = self.anomalous_location + self.start_idx

        bin_df = df.iloc[self.anomaly_index_raw]
        bin_df.insert(
            len(bin_df.keys()),
            "Bin Score",
            self._filtered_zscore[self.anomalous_location],
        )
        self.bin_df = bin_df
        self.logger.debug("HSA model predictions complete.")
        self.preprocessed_df = df

    def global_collect_multifilter_df(
        self,
        total_preprocessed_np: np.ndarray,
        total_anomaly_index: list,
        mf_batch_size: int,
        _ones: np.array,
    ):
        """Given a fixed set of anomalies collects random background pixels
        from throughout the entire preprocessed_np"""
        assert isinstance(
            total_preprocessed_np, np.ndarray
        ), f"total_preprocessed_np must be a np.ndarray, not {type(total_preprocessed_np)}"
        assert isinstance(
            _ones, np.ndarray
        ), f"_ones must be a np.ndarray, not {type(_ones)}"
        assert isinstance(
            mf_batch_size, int
        ), f"mf_batch_size must be an int, not {type(mf_batch_size)}"

        prob_vec = _ones
        prob_vec[total_anomaly_index] = 0
        prob_vec = prob_vec / sum(prob_vec)

        padding_index = np.random.choice(
            len(total_preprocessed_np), mf_batch_size, p=prob_vec
        )  # random select non-anomalous data from current preprocessed_np
        mix_data = np.append(
            total_preprocessed_np[total_anomaly_index],
            total_preprocessed_np[padding_index],
            axis=0,
        )

        mix_index = np.append(total_anomaly_index, padding_index)

        self.logger.trace("Global Multifilter Complete.")
        return mix_index, mix_data, total_anomaly_index

    def apf_df_generation(self, total_anomaly_index):
        self.anomaly_prediction_frequency_df = pd.DataFrame()
        self.anomaly_prediction_frequency_df["User DF Index"] = total_anomaly_index
        self.anomaly_prediction_frequency_df.set_index("User DF Index")
        self.anomaly_prediction_frequency_df["Anomaly Bin Count"] = np.zeros(
            len(total_anomaly_index)
        )
        self.total_anomaly_index = total_anomaly_index
        return self.anomaly_prediction_frequency_df

    def uni_shuffle_multifilter_df(
        self,
        mix_index: np.ndarray,
        mix_data: np.ndarray,
        predicted_anomaly: np.ndarray,
    ):
        """Works in conjunction with the collect_multifilter_df methods
        by randomly shuffling the anomalies and background data and indices
        in unison for later recall to raw data"""

        assert isinstance(
            mix_index, np.ndarray
        ), f"mix_index must be an np.ndarray, not {type(mix_index)}"
        assert isinstance(
            mix_data, np.ndarray
        ), f"mix_index must be an np.ndarray, not {type(mix_data)}"
        assert isinstance(
            predicted_anomaly, np.ndarray
        ), f"mix_data must be an np.ndarray, not {type(predicted_anomaly)}"
        ## shuffle after all data is collected
        self.shuffler = np.random.permutation(
            len(mix_index)
        )  # generate a shuffle in unison for index and data
        self.all_index_user = mix_index[self.shuffler]  # shuffle index
        self.all_data = mix_data[self.shuffler]  # same shuffle for data

        self.current_anomaly_index = np.where(
            np.isin(self.all_index_user, predicted_anomaly)
        )
        self.all_index_mf = self.all_index_user - self.start_idx
        self.logger.trace("Multifilter data has been shuffled.")
        return self

# HSA Pipeline Documentation
This document details the `HSA_pipeline` class, which transforms preprocessed data through the complete anomaly detection workflow from input to inference.

<sub><sub><sub>Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED</sub></sub></sub>

## HSA_pipeline Class

### Constructor

```python
class HSA_pipeline():
    def __init__(
        self,
        bin_count: int,
        min_additional_percent_variance_exp: int,
        max_spawn_dummies: int,
        anomaly_std_tolerance: float,
        cutoff_distance: float,
        lr: float,
        penalty_ratio: float,
        percent_variance_explained: float,
        base_directory: str,
        logger: loguru_logger,
        logging_level: str,
        multi_filters: int = 15,
        affinity_matrix_iterations: int = 20,
        batch_size: int = 2000,
        iterations: int = int(5e3),
        num_workers: int = 10,
        converge_toll: float = 1e-30,
        plot_figures: bool = False,
        save_figures: bool = False,
        save_preprocessed_np: bool = False,
        verbose: bool = False,
    ):
```

### Core Parameters

**Detection Parameters**
- **bin_count**: Multifilter threshold for anomalous prediction counts
- **anomaly_std_tolerance**: Model sensitivity for anomaly selection (standard deviations from mean)
- **cutoff_distance**: Hyperparameter for similarity and density calculations
- **penalty_ratio**: Decay rate for topographical scale contributions in objective function

**Preprocessing Parameters**
- **min_additional_percent_variance_exp**: Minimum variance contribution for PCA feature retention
- **max_spawn_dummies**: Maximum features allowed from categorical encoding
- **percent_variance_explained**: PCA stopping tolerance for feature selection

**Optimization Parameters**
- **lr**: Learning rate for PyTorch Adam optimizer
- **iterations**: Number of optimization iterations (default: 5,000)
- **converge_toll**: Convergence tolerance for anomaly score vector (default: 1e-30)
- **affinity_matrix_iterations**: Number of topological scales in objective function (default: 20)

**Processing Parameters**
- **batch_size**: Data loader batch size (default: 2,000)
- **multi_filters**: Number of multifilter iterations (default: 15)
- **num_workers**: Available worker threads (default: 10)

**Output Parameters**
- **base_directory**: Storage location for results/, logs/, and plots/
- **logger**: Loguru logger instance
- **logging_level**: Logger verbosity level
- **plot_figures**: Display weight and prediction heatmaps
- **save_figures**: Save figures to {base_directory}/plots/
- **save_preprocessed_np**: Save processed arrays to {base_directory}/results/
- **verbose**: Print logged statements to terminal

## Pipeline Method

### Parameters

- **df**: Input DataFrame containing the data for anomaly detection
- **if_preprocess**: Apply StandardScaler and PCA preprocessing (requires variance and dummy parameters)

### Returns

- **results_df**: Original input data with additional anomaly score column
- **logging**: Detailed log file saved as {logging_directory}/HSA_{run_date}.log

### Workflow

The pipeline method orchestrates the complete anomaly detection process:

1. **Preprocessing** (if enabled): Applies categorical encoding, scaling, and PCA feature selection
2. **Model Training**: Constructs similarity matrices and evolves affinity relationships
3. **Optimization**: Minimizes penalized objective function to compute anomaly scores
4. **Multifilter**: Reduces false positives through iterative re-evaluation
5. **Results**: Returns annotated DataFrame with anomaly classifications

The method handles batch processing automatically and provides comprehensive logging throughout the detection workflow.

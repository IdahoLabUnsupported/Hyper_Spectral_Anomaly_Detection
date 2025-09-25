# Hyper Spectral Anomaly Detection (HSA)
An unsupervised anomaly detection model for identifying outliers in tabular data using graph-based similarity analysis. Applicable to diverse data types—such as cybersecurity, materials testing, and GIS—providing unsupervised detection of unusual patterns while reducing false positives with a multi-filter process.

<sub><sub><sub>Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED</sub></sub></sub>


## Quick Start

```python
import HSA_Classes.HSA_Pipeline as HSA_pipe
import pandas as pd
from loguru import logger

# Load your preprocessed data
data = pd.read_pickle("data.pkl")

# Initialize the model with default parameters
model = HSA_pipe.HSA_pipeline(
    data,
    penalty_ratio=0.75,
    cutoff_distance=3,
    lr=2.73,
    anomaly_std_tolerance=1.5,
    bin_count=14,
    max_spawn_dummies=500,
    percent_variance_explained=0.95,
    min_additional_percent_variance_exp=0.005,
    logger=logger,
    logging_level="INFO"
)

# Run anomaly detection
results_df = model.infer()
```

## Key Parameters

### Preprocessing
- **max_spawn_dummies** (default: 500): Upper limit on encoding space size per feature
- **percent_variance_explained** (default: 0.95): PCA variance threshold for feature selection
- **min_additional_percent_variance_exp** (default: 0.005): Minimum variance for keeping features

### Model Configuration
- **penalty_ratio** (default: 0.75): Controls model generalizability vs specificity
- **cutoff_distance** (default: 3): Similarity threshold for data relationships
- **lr** (default: 2.73): Learning rate for optimization

### Detection Thresholds
- **anomaly_std_tolerance** (default: 1.5): Standard deviations from mean to classify as anomaly
- **bin_count** (default: 14): Minimum prediction count in multifilter stage

## Model Architecture

The HSA pipeline consists of three main components:

### Preprocessing
Handles categorical and object data encoding, converts time-like objects to PyTorch-compatible data types, and scales the data. Uses PCA via `sklearn.decomposition` to select relevant features based on variance explained thresholds. Returns a pandas DataFrame of raw data for reporting, a scaled/encoded/PCA numpy array for model training, and fitted sklearn decomposer and scaler objects.

### HSA_Model  
The core anomaly detection class that generates model weights through vertex weight calculations and distance computations. Deviates from the original paper's approach by using matrix operations instead of loops for significant performance improvements. Creates affinity matrices, similarity matrices, and other mathematical constructs, then evolves these matrices through different powers to analyze relationships across multiple topological scales. Uses PyTorch's Adam optimizer to minimize the penalized objective function and determine anomaly scores.
### Multi-Filter Approach
To reduce false positives, the model employs a multi-filter strategy:

1. Collect all anomalous predictions from individual batches
2. Create a balanced dataset (10% anomalous, 90% normal) 
3. Re-run detection on this mixed dataset multiple times
4. Count prediction frequency for each potential anomaly
5. Filter out inconsistent predictions based on `bin_count` threshold

This approach allows local anomalies to be compared against the global dataset, reducing false positives from batch-specific artifacts. 

### Visualization
Provides model explainability through visualization tools. The `heatmap_weights_matrix` method generates heatmaps of model weights, matrices, and preprocessed data to show how anomalous events create "hot spots" that propagate through optimization. The `heatmap_bin_predictions` method creates visualizations of multifilter data, anomaly score locations, and score values for visual inspection of results.

## How It Works

HSA detects anomalies by:
1. Computing similarity relationships between data points using Euclidean distances
2. Building affinity matrices that capture multi-scale topological relationships
3. Optimizing anomaly scores through a penalized objective function
4. Using a multifilter approach to reduce false positives

The model is particularly effective at finding anomalies that are dissimilar in feature space rather than just geometric proximity.

## Examples

See `Documentation/Examples/` for detailed usage examples and tutorials.

## Output

The model returns a dataframe containing:
- Raw data for detected anomalous events
- Anomaly scores
- Prediction confidence metrics from the multifilter stage

## Background

This implementation is based on graph evolution techniques from hyperspectral image analysis, adapted for general tabular anomaly detection. The approach uses graph theory principles to analyze data relationships across multiple topological scales.

---

## Technical Details & Mathematical Foundation

### Motivation

The HSA model is based on "Graph Evolution-Based Vertex Extraction for Hyperspectral Anomaly Detection" by Xianchang et al. Originally designed for hyperspectral image analysis, this method has been adapted for general anomaly detection by comparing data points in function space rather than focusing solely on geometric proximity.

### Mathematical Framework

#### Distance and Edge Weight Calculation

For data $\vec{X}\in \mathbb{R}^{n,m}$ with $n$ observations and $m$ features, we calculate pairwise Euclidean distances:

$d_{ij}=||\vec{x_i}-\vec{x_j}||$

Edge weights use a Gaussian radial basis function:

$\gamma_i=\sum_{i\neq j}e^{-\left( d_{ij}/d_c\right)^2} \quad:\quad\vec{\gamma}\in \mathbb{R}^n$

where $d_c$ is the cutoff distance hyperparameter.

#### Similarity Matrix Construction

The similarity matrix $\vec{S}\in \mathbb{R}^{n,n}$ encodes separation information:

$s_{ij}=e^{\left(- d_{ij}/d_c^2 \right)}$

This is normalized using diagonal matrix $\vec{D}\in \mathbb{R}^{n,n}$:

$d_{i=j}=\left( \left(\sum_j^n s_{ij}\right)^{-1/2} \right) \quad d_{i\neq j}=0$

The normalized similarity matrix is:

$\tilde{S}=\vec{D}\vec{S}\vec{D}$

#### Affinity Matrix Generation

The affinity matrix $\vec{A}\in \mathbb{R}^{n,n}$ combines density and similarity information:

$\vec{A}=\vec{\Gamma}\tilde{S}\vec{\Gamma}$

where $\vec{\Gamma} \in \mathbb{R}^{n,n}$ is the diagonal matrix with ${\gamma_{i=j}'}={\gamma_i}$ and $\gamma_{i\neq j}=0$.

#### Graph Evolution

To analyze relationships across multiple topological scales, we compute matrix powers:

$\mathbf{A}=\set{\vec{A}, \vec{A}^2,\vec{A}^3,\dots,\vec{A}^k}$

and 

$\mathbf{D}=\set{\vec{D}, \vec{D}^2,\vec{D}^3,\dots,\vec{D}^k}$

#### Penalized Objective Function

Anomaly scores $\vec{m}\in\mathbb{R}^{n}$ are found by minimizing:

$f\left(\vec{m}\right)=\frac{1}{2}\vec{m}^T\vec{A}^k\vec{m}+\left(\vec{\gamma}^T\vec{D}\right)^T\vec{m}$

With constraints $0\le {m_i}\le1$ enforced via penalty function:

$\vec{p}(\vec{m_p})=r^k\vec{m_p}^q$

where $q \in \mathbb{Z}:q \pmod{2}=0$ and:

$$\vec{m_p} = \begin{cases} 
0 & \text{for } 0 \leq m_i \leq 1 \\
-m_i & \text{for } m_i \lt 0 \\
m_i - 1 & \text{for } m_i \gt 1
\end{cases}$$

The complete objective function ${\Phi}(\vec{m})\equiv f(\vec{m})+\sum_{i}^n {p_i}(\vec{m})$ is minimized using PyTorch's Adam optimizer.

#### Anomaly Threshold

Anomaly scores are converted to binary classifications using standard deviation thresholds. Points with scores exceeding `anomaly_std_tolerance` standard deviations from the mean are classified as anomalous.


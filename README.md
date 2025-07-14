# Clinical Trajectory Patterns in IgA Nephropathy Identified by Deep Learning

This repository contains the code for the analysis presented in the paper "Clinical Trajectory Patterns in IgA Nephropathy Identified by Deep Learning and Their Association with Renal Outcomes."

## Project Overview

This project uses a Long Short-Term Memory (LSTM) autoencoder to identify distinct clinical trajectory patterns in patients with IgA nephropathy. The identified patient clusters are then analyzed for their association with long-term renal outcomes.

## Getting Started

### Prerequisites

- Python 3.9+
- Pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

This repository does not include the clinical data used in the study. To use this code with your own data, you will need to prepare three files in the `data/` directory:

1.  **`seq_tensor.npz`**: A NumPy archive containing the time-series data.
    -   `X`: A 3D NumPy array of shape `(n_patients, n_timesteps, n_features)`.
    -   `patient_ids`: A 1D NumPy array of patient identifiers.

2.  **`baseline.csv`**: A CSV file with baseline patient characteristics.
    -   Must contain a `patient_id` column corresponding to the IDs in `seq_tensor.npz`.

3.  **`outcome.csv`**: A CSV file with the outcome data for survival analysis.
    -   Must contain a `patient_id` column.
    -   Must contain columns for `time_to_event` and `event`.

## Running the Analysis

The analysis is divided into two main scripts:

1. **`01_train_lstm_autoencoder.py`**: This script trains the LSTM autoencoder on the time-series data and saves the trained model and the extracted latent representations.

   ```bash
   python 01_train_lstm_autoencoder.py
   ```

2. **`02_analyze_clusters.py`**: This script loads the latent representations, performs K-means clustering, and generates the survival analysis plots and other visualizations.

   ```bash
   python 02_analyze_clusters.py
   ```

## Output

The following outputs will be generated in the `output/` directory:

- **`output/models/`**: Contains the trained LSTM autoencoder model (`lstm_autoencoder.h5`).
- **`output/latent_representations.npz`**: The latent space representations of the patient trajectories.
- **`output/figures/`**: Contains the generated plots, including the t-SNE visualization and Kaplan-Meier curves.
- **`output/tables/`**: Contains the final clustered patient data (`clustered_patient_data.csv`).

## Citation

If you use this code in your research, please cite the original paper:

> Noda, R., Ichikawa, D., Shirai, S., & Shibagaki, Y. (2025). Clinical Trajectory Patterns in IgA Nephropathy Identified by Deep Learning and Their Association with Renal Outcomes. 

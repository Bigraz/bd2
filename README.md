# Assignment 2 — Predicting GitHub User Activity

This project uses a two-layer graph convolutional network (GCN) to predict whether a GitHub user will perform each of six activities during the following week. A user can receive multiple positive predictions.

## Data Source

The data comes from [GH Archive](https://www.gharchive.org/), which publishes hourly files of public GitHub events. The reported experiment used archive data from October 1 to November 8, 2026. All model inputs are derived from the eight anonymized tables built by the [official assignment preprocessing script](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2).

## 1. Prepare the data

Run the following from this repository's root in Bash. If the course scripts are missing after cloning this project, fetch the course repository first:

```bash
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git BigData2026-Assignment2
```

Download the hourly archives and preprocess them using Python, pandas, and PyArrow. Keep the supplied `cohort.txt` next to `preprocess.py`.

```bash
python -m pip install pandas pyarrow
mkdir -p raw
wget -nc -P raw https://data.gharchive.org/2026-10-{01..31}-{0..23}.json.gz
wget -nc -P raw https://data.gharchive.org/2026-11-{01..08}-{0..23}.json.gz
python BigData2026-Assignment2/scripts/preprocess.py --raw raw --out BD_LAB2
```

Copy the resulting `BD_LAB2/db/` directory to `MyDrive/BD_LAB2/db/` in Google Drive. Also copy the official `scripts/evaluate.py` and `submission/test_input.csv` into the corresponding locations below:

```text
MyDrive/BD_LAB2/
├── db/                       # Eight Parquet tables
├── scripts/evaluate.py
└── submission/test_input.csv
```

## 2. Train the model

Open [lab2_training_final.ipynb](my_files/lab2_training_final.ipynb) in Google Colab and run the cells in order. The notebook installs PyTorch Geometric and PyArrow, mounts Drive, and uses PyTorch with CUDA when available.

The training cutoff is October 25, 2026 at 00:00 UTC. Earlier records supply the graph edges and historical features. Activity during October 25–31 supplies the six binary training labels per target user.

- **Graph:** Seven tables supply nodes. Push records connect users directly to repositories. Foreign-key links and their reverse edges connect the remaining records.
- **Features:** Each node has seven node-type indicators. User nodes also receive twelve log-transformed counts covering six activities over the preceding 7 and 30 days. The training cutoff has 24 days of available history for the nominal 30-day window.
- **Training:** The GCN maps 19 input features to 64 hidden values and then six logits. Training uses ReLU, dropout, Adam (learning rate 0.001), and 100 epochs. The loss is binary cross-entropy with a separate positive weight for each activity based on its negative-to-positive label ratio.

The notebook writes the trained weights and configuration to `MyDrive/BD_LAB2/training_outputs/gcn_checkpoint.pt`. Retraining can produce different weights because the training notebook does not set a fixed random seed.

## 3. Validate on a later week

Open [lab2_evaluation.ipynb](my_files/lab2_evaluation.ipynb). Keep the November 1 cutoff and **set `threshold = 0.70`** in the configuration cell before running all cells. The notebook loads the checkpoint, rebuilds the inputs at the new cutoff, and evaluates against observed activity during November 1–7.

Predictions, sigmoid scores, ground-truth labels, and the evaluator's results are saved under `MyDrive/BD_LAB2/evaluation_outputs/`. The course score is the mean of macro-F1 and micro-F1. On the reported validation run, increasing the threshold from 0.50 to 0.70 increased this score from approximately 20% to 40%.

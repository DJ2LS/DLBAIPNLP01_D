# Predicting Star Ratings of Amazon Reviews with BERT

Project DLBAIPNLP01_D (NLP, task 1): A fine-tuned `bert-base-uncased` model predicts the rating (1–5 stars) of an Amazon review from its title and text. A second model predicts the polarity (negative, neutral, positive).

- **Data:** [Amazon Reviews'23](https://amazon-reviews-2023.github.io/), categories Software, Cell_Phones_and_Accessories, Video_Games, Electronics
- **Sample:** 230,000 English reviews per category and star rating, split 80/10/10 (train/val/test)

## Requirements

- Python 3.12 or newer (developed with 3.14)
- about 15 GB of free disk space for the raw and processed data
- a CUDA GPU for training, either locally or in the cloud via [Modal](https://modal.com) (step 5)

## Installation (Linux)

```bash
git clone https://github.com/DJ2LS/DLBAIPNLP01_D.git
cd DLBAIPNLP01_D
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To train on Modal, log in once:

```bash
modal setup
```

## Usage

Run all scripts from the project folder in order; each script reads the output of the previous one.

| Step | Command | Output |
|---|---|---|
| 1. Download data | `python 1_download_reviews.py` | `data/*.jsonl.gz` (approx. 9.6 GB) |
| 2. Count ratings | `python 2_rating_stats.py` | `results/rating_distribution_full.csv` |
| 3. Clean, sample, split | `python 3_prepare_splits.py` | `data/splits/{train,val,test}.csv`, `results/removed_reviews.csv`, `results/split_overview.csv` |
| 4. Preprocess text | `python 4_preprocess_text.py` | `data/processed/{train,val,test}.csv`, `results/preprocessing_stats.csv` |
| 5. Train BERT | `modal run 5_train_bert_modal.py` (cloud) or `python 5_train_bert.py` (local GPU) | `results/bert_log_*.csv`, `results/predictions_*.csv` |
| 6. Evaluate | `python 6_evaluate.py` | `results/test_results.csv`, `results/test_accuracy_per_category.csv`, `results/confusion_matrix_*.csv` |

## Results without retraining

The training results are already included in `results/`. To only repeat the evaluation, run this after the installation:

```bash
python 6_evaluate.py
```

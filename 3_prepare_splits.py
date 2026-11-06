"""Removes reviews without text and non-English reviews, samples 230,000 reviews per category and star rating and
splits them 80/10/10. Output: data/splits/{train,val,test}.csv, results/split_overview.csv, results/removed_reviews.csv"""

from pathlib import Path

import numpy as np
import pandas as pd
from lingua import Language, LanguageDetectorBuilder

PER_STAR = 230000
SPLITS = ["train", "val", "test"]
SEED = 26

rng = np.random.default_rng(SEED)
totals = pd.read_csv("results/rating_distribution_full.csv", index_col="category")
# Language: test against all 75 languages supported by lingua, keep only reviews whose most likely language is English
detector = LanguageDetectorBuilder.from_all_languages().build()

samples = []
removed = {}
removed_not_english = {}
for path in sorted(Path("data").glob("*.jsonl.gz")):
    category = path.name.removesuffix(".jsonl.gz")
    print(f"Sampling {category} …")
    keep_prob = {s: min(1.0, 1.2 * PER_STAR / totals.loc[category, f"{s}_star"]) for s in range(1, 6)}

    kept = []
    removed[category] = 0
    for chunk in pd.read_json(path, lines=True, chunksize=500000, dtype={"rating": int}):
        chunk = chunk[["rating", "title", "text"]]
        # Remove reviews without text
        empty = chunk["text"].str.strip() == ""
        removed[category] += int(empty.sum())
        chunk = chunk[~empty]
        kept.append(chunk[rng.random(len(chunk)) < chunk["rating"].map(keep_prob)])

    kept = pd.concat(kept)
    language = detector.detect_languages_in_parallel_of((kept["title"] + ". " + kept["text"]).tolist())
    english = pd.Series([lang == Language.ENGLISH for lang in language], index=kept.index)
    removed_not_english[category] = int((~english).sum())
    kept = kept[english]

    sample = kept.groupby("rating").sample(n=PER_STAR, random_state=SEED)
    samples.append(sample.assign(category=category))

data = pd.concat(samples, ignore_index=True)[["category", "rating", "title", "text"]]
data = data.sample(frac=1, random_state=SEED)
position = data.groupby(["category", "rating"]).cumcount() / PER_STAR
data["split"] = np.select([position < 0.8, position < 0.9], ["train", "val"], "test")

Path("data/splits").mkdir(parents=True, exist_ok=True)
for name in SPLITS:
    data[data["split"] == name].drop(columns="split").to_csv(f"data/splits/{name}.csv", index=False)

overview = pd.crosstab([data["split"], data["category"]], data["rating"], margins=True, margins_name="Summe")
overview = overview.reindex(SPLITS + ["Summe"], level=0)
removed_reviews = pd.DataFrame({"total": totals["total"], "removed_empty_text": pd.Series(removed),
                                "removed_not_english": pd.Series(removed_not_english)})
print(overview.to_string())
print(removed_reviews.to_string())

overview.to_csv("results/split_overview.csv")
removed_reviews.to_csv("results/removed_reviews.csv", index_label="category")

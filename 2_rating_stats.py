"""Counts the star ratings across all reviews in data/. Output: results/rating_distribution_full.csv"""

from pathlib import Path
import pandas as pd

counts = {}
for path in sorted(Path("data").glob("*.jsonl.gz")):
    print(f"Counting {path.name} …")
    chunks = pd.read_json(path, lines=True, chunksize=500000)
    ratings = pd.concat(chunk["rating"].astype("int8") for chunk in chunks)
    counts[path.name.removesuffix(".jsonl.gz")] = ratings.value_counts()

counts = pd.DataFrame(counts).T[[1, 2, 3, 4, 5]].astype(int).rename_axis("category")
counts.columns = [f"{s}_star" for s in counts.columns]
counts["total"] = counts.sum(axis=1)

print(counts.to_string())
print((counts.drop(columns="total").div(counts["total"], axis=0) * 100).round(1).to_string())

Path("results").mkdir(exist_ok=True)
counts.to_csv("results/rating_distribution_full.csv")

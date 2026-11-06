"""Evaluates the BERT predictions on the test set (accuracy, macro-F1, ±1 star, MAE, confusion matrix,
per category, rating->polarity). Output: results/test_results.csv, test_accuracy_per_category.csv,
confusion_matrix_<task>.csv"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

POLARITY = {1: "negativ", 2: "negativ", 3: "neutral", 4: "positiv", 5: "positiv"}
CLASSES = {"rating": [1, 2, 3, 4, 5], "polarity": ["negativ", "neutral", "positiv"]}

rating = pd.read_csv("results/predictions_rating.csv")
polarity = pd.read_csv("results/predictions_polarity.csv")
rating_as_polarity = rating.assign(true=rating["true"].map(POLARITY), prediction=rating["prediction"].map(POLARITY))
error = (rating["true"] - rating["prediction"]).abs()


def scores(df):
    return {"accuracy": round(accuracy_score(df["true"], df["prediction"]), 4),
            "macro_f1": round(f1_score(df["true"], df["prediction"], average="macro"), 4)}


results = pd.DataFrame({
    "rating (5 classes)": {**scores(rating), "accuracy_within_1": round((error <= 1).mean(), 4),
                           "mae": round(error.mean(), 4)},
    "polarity (3 classes)": scores(polarity),
    "rating -> polarity mapped": scores(rating_as_polarity),
}).T

per_category = pd.DataFrame({
    name: df.assign(correct=df["true"] == df["prediction"]).groupby("category")["correct"].mean().round(4)
    for name, df in {"rating": rating, "polarity": polarity}.items()
})

for task, df in {"rating": rating, "polarity": polarity}.items():
    cm = pd.DataFrame(confusion_matrix(df["true"], df["prediction"], labels=CLASSES[task]),
                      index=[f"true_{c}" for c in CLASSES[task]], columns=[f"pred_{c}" for c in CLASSES[task]])
    cm.to_csv(f"results/confusion_matrix_{task}.csv")
    print(f"\nConfusion matrix {task}:\n{cm.to_string()}")

print(f"\n{results.to_string()}\n\n{per_category.to_string()}")
results.to_csv("results/test_results.csv", index_label="system")
per_category.to_csv("results/test_accuracy_per_category.csv")

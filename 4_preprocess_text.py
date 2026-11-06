"""Prepares the texts for BERT (title + text, HTML entities and emojis written out, HTML tags, video placeholders
and URLs removed) and separates input (text) from output (rating, polarity).
Output: data/processed/{train,val,test}.csv, results/preprocessing_stats.csv"""

import html
from pathlib import Path

import emoji
import pandas as pd
from transformers import AutoTokenizer

MAX_LENGTH = 256
POLARITY = {1: "negativ", 2: "negativ", 3: "neutral", 4: "positiv", 5: "positiv"}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
Path("data/processed").mkdir(parents=True, exist_ok=True)

stats = []
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/{split}.csv", keep_default_na=False)
    text = df["title"] + ". " + df["text"]
    # Write out HTML entities: &#34; -> ", &amp; -> &
    text = text.map(html.unescape)
    # Remove HTML tags: <br />
    text = text.str.replace(r"<[^>]+>", " ", regex=True)
    # Remove video placeholders: [[VIDEOID:8f82bce0…]]
    text = text.str.replace(r"\[\[VIDEOID:\w+\]\]", " ", regex=True)
    # Shorten product references to the product name
    text = text.str.replace(r"\[\[ASIN:\w+ ?([^\]]*)\]\]", r"\1", regex=True)
    # Remove URLs
    text = text.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    # Write out emojis: 👍 -> thumbs up
    text = text.map(lambda t: emoji.replace_emoji(t, replace=lambda e, d: f" {d['en'].strip(':').replace('_', ' ')} "))
    # Collapse multiple spaces and line breaks
    text = text.str.split().str.join(" ")
    pd.DataFrame({"category": df["category"], "text": text, "rating": df["rating"],
                  "polarity": df["rating"].map(POLARITY)}).to_csv(f"data/processed/{split}.csv", index=False)

    n_tokens = text.map(lambda t: len(tokenizer(t, verbose=False)["input_ids"]))
    stats.append({"split": split, "reviews": len(df),
                  "avg_words": round(text.str.split().str.len().mean(), 1),
                  "avg_bert_tokens": round(n_tokens.mean(), 1),
                  "median_bert_tokens": int(n_tokens.median()),
                  f"share_over_{MAX_LENGTH}_tokens_%": round((n_tokens > MAX_LENGTH).mean() * 100, 1)})

stats = pd.DataFrame(stats).set_index("split")
print(stats.to_string())
stats.to_csv("results/preprocessing_stats.csv")

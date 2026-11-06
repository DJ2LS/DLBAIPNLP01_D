"""Fine-tunes BERT twice: on the rating (5 classes) and on the polarity (3 classes).
Output: results/bert_log_<task>.csv (training history), results/predictions_<task>.csv (test set)"""

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
EPOCHS = 2
BATCH_SIZE = 512
LEARNING_RATE = 5e-5
SEED = 26
TASKS = {"rating": [1, 2, 3, 4, 5], "polarity": ["negativ", "neutral", "positiv"]}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**tokenizer(self.texts[i], truncation=True, max_length=MAX_LENGTH), "labels": self.labels[i]}


def accuracy(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (np.argmax(logits, axis=1) == labels).mean()}


splits = {s: pd.read_csv(f"data/processed/{s}.csv") for s in ["train", "val", "test"]}
X = {s: df["text"].tolist() for s, df in splits.items()}

for task, classes in TASKS.items():
    print(f"\n===== {task} ({len(classes)} classes) =====")
    y = {s: df[task].map({c: i for i, c in enumerate(classes)}).tolist() for s, df in splits.items()}
    datasets = {s: ReviewDataset(X[s], y[s]) for s in splits}

    args = TrainingArguments(
        output_dir=f"bert_checkpoints/{task}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        auto_find_batch_size=True,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=0.1,
        train_sampling_strategy="group_by_length",
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        dataloader_num_workers=8,
        report_to="none",
        seed=SEED,
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(classes))
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["val"],
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=accuracy)
    trainer.train()
    pd.DataFrame(trainer.state.log_history).to_csv(f"results/bert_log_{task}.csv", index=False)

    # Predict the test set in its original order: in transformers 5, group_by_length would reorder it as well
    trainer.args.train_sampling_strategy = "sequential"
    predicted = np.argmax(trainer.predict(datasets["test"]).predictions, axis=1)
    test = splits["test"]
    pd.DataFrame({"category": test["category"], "true": test[task],
                  "prediction": [classes[i] for i in predicted]}).to_csv(f"results/predictions_{task}.csv", index=False)

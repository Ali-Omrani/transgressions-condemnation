import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Dataset

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pickle
import pandas as pd
from datasets import Value
from torch.utils.data import DataLoader
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model-path', default="")
    parser.add_argument('--prediciton-save-dir', default="results")
    parser.add_argument('--pretrained-model', default="bert-base-uncased")
    parser.add_argument('--pred-data-path', default="../../data/5_mil_7days_metoo.p")
    args = parser.parse_args()

    checkpoint = args.pretrained_model
    with open(args.pred_data_path, 'rb') as f:
        pred_data = pickle.load(f)

    pred_df = pred_data[["clean_tweet_masked"]].dropna()
    pred_dataset = Dataset.from_pandas(pred_df)
    pred_dataset = pred_dataset.rename_column("clean_tweet_masked", "text")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_datasets = pred_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_path = args.model_path
    model = torch.load(model_path)
    model.eval()

    args = TrainingArguments(
        output_dir="exp/bart/results",
        do_train=False,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=1,
    )
    training_args = TrainingArguments("test-trainer")
    training_args.eval_accumulation_steps = 1  # pushes predictions out of GPU to mitigate GPU out of memory

    trainer = Trainer(
        model,
        args=args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    predictions = trainer.predict(tokenized_datasets)

    with open(os.path.join(args.prediciton_save_dir, 'condemnation_predictions.p', 'wb')) as f:
        pickle.dump(predictions, f)


if __name__ == "__main__":
    main()
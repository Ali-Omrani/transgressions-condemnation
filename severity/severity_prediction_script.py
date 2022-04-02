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
from pymongo import MongoClient
import IPython
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model-path', default="")
    parser.add_argument('--prediciton-save-dir', default="results")
    parser.add_argument('--pretrained-model', default="bert-base-uncased")
    parser.add_argument('--pred-data-path', default="./temp/to_predict.p")
    args = parser.parse_args()

    checkpoint = args.pretrained_model
    model_path = args.model_path
    model = torch.load(model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="exp/bart/results",
        do_train=False,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=1,
    )
    # training_args = TrainingArguments("test-trainer")
    # training_args.eval_accumulation_steps = 1  # pushes predictions out of GPU to mitigate GPU out of memory

    trainer = Trainer(
        model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    for pred_file in os.listdir(args.pred_data_path):
        if os.path.isdir(os.path.join(args.pred_data_path, pred_file)):
            print("skipping directory {}".format(pred_file))
            continue

        save_file_path = os.path.join(args.prediciton_save_dir, 'condemnation_prediction_chunk_'+ pred_file.split("_")[-1])

        if os.path.exists(save_file_path):
            print("predictions for {} already exits".format(save_file_path))
            continue

        print("working on", pred_file)
        with open(os.path.join(args.pred_data_path, pred_file), 'rb') as f:
            pred_data = pickle.load(f)

        pred_df = pred_data[["clean_tweet_masked"]].dropna()
        pred_data = pred_data.dropna(subset=["clean_tweet_masked"])
        pred_dataset = Dataset.from_pandas(pred_df)
        pred_dataset = pred_dataset.rename_column("clean_tweet_masked", "text")

        tokenized_datasets = pred_dataset.map(tokenize_function, batched=True)


        predictions_logits = trainer.predict(tokenized_datasets)
        preds = np.argmax(predictions_logits.predictions, axis = 1)

        save_df = pred_data[["_id"]]
        save_df["severity_prediction"] = preds
        save_df["severity_logit_0"] = predictions_logits.predictions[:, 0]
        save_df["severity_logit_1"] = predictions_logits.predictions[:, 1]
        save_df["severity_logit_2"] = predictions_logits.predictions[:, 2]
        IPython.embed(); exit();
        with open(save_file_path, 'wb') as f:
            pickle.dump(save_df, f)




if __name__ == "__main__":
    # params = --model-path ./models/fold_1_model.p --pred-data-path ../condemnation/results
    os.listdir("../condemnation/results"))
    IPython.embed(); exit();
    # main()

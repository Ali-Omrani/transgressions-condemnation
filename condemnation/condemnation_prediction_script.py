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
        save_df["condemnation_prediction"] = preds
        save_df["condemnation_logit_0"] = predictions_logits.predictions[:, 0]
        save_df["condemnation_logit_1"] = predictions_logits.predictions[:, 1]

        with open(os.path.join(args.prediciton_save_dir, 'condemnation_prediction_chunk_'+ pred_file.split("_")[-1]), 'wb') as f:
            pickle.dump(save_df, f)

def get_prediction_dataframe(db_name = "new_metoo", query = {"is_RT": True}):
    def split_list(cursor, n):
        result = []
        for tweet in tqdm(cursor):
            result.append(tweet)
            if len(result)==n:
                result_to_return = result
                result = []
                yield result_to_return
        yield result

    client = MongoClient()
    db_metoo_tweets = client[db_name]
    metoo_tweets = db_metoo_tweets.metoo_tweets
    cursor = metoo_tweets.find(query)
    print("got cursor")
    # list_cur = list(cursor)
    # print("listed cursor")
    for idx, chunk in tqdm(enumerate(split_list(cursor, 100000))):
        df = pd.DataFrame(chunk)
        with open("./temp/pred_chunk_{}.p".format(idx), "wb") as f:
             pickle.dump(df, f)


def push_predictions_to_db(db_name = "new_metoo", pred_data_path = "./results" ):
    client = MongoClient()
    db_metoo_tweets = client[db_name]
    metoo_tweets = db_metoo_tweets.metoo_tweets
    def update_tweet_in_db( document):
        try:
            metoo_tweets.update_one(
                {'_id': document['_id']},
                {'$set': document}
            )
        except Exception:
            print("couldn't update ", document)


    for i, pred_file in enumerate(os.listdir(pred_data_path)):
        print(i, "files pushed to DB")
        if os.path.isdir(os.path.join(pred_data_path, pred_file)):
            print("skipping directory {}".format(pred_file))
            continue

        with open(os.path.join(pred_data_path, pred_file), 'rb') as f:
            pred_data = pickle.load(f)
        for idx, row in tqdm(pred_data.iterrows()):
            update_tweet_in_db(row.to_dict())



# print("df generated")
    # if not os.path.exists("./temp"):
    #     os.mkdir("temp")
    # print("saved df")
    # IPython.embed();
    # exit();



if __name__ == "__main__":
    # params = --model-path ./models/fold_1_model.p --pred-data-path ./temp/
    # get_prediction_dataframe()
    # main()
    push_predictions_to_db()
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from models.config import BERT_MODEL_NAME
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
import json
import shutil


class WeightedLossTrainer(Trainer):
    def __init__(self, classes_weights=torch.tensor([1, 1]), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_weights = classes_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.classes_weights)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class MultiLabelTrainer(WeightedLossTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.BCEWithLogitsLoss(weight=self.classes_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class BertBase:
    def __init__(self, model_name=BERT_MODEL_NAME, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def _create_tokenized_dataset(self, df):
        dataset = Dataset.from_pandas(df[['text', 'labels']])
        dataset = dataset.map(self._preprocess_function, batched=True)
        return dataset

    def _create_trainer(self, CustomTrainer, model, compute_metrics, train_data=None, val_data=None,
                        classes_weights=None, output_dir=None, num_train_epochs=5):
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model='eval_F1',
            greater_is_better=True,
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator,
            classes_weights=classes_weights
        )
        return trainer

    def load_model(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path)


class BertClassifier(BertBase):
    def cross_validation(self, df, output_dir):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        classes_weights = torch.tensor(
            list(df.groupby('labels')['labels'].count() / len(df))[::-1]).to(device)

        dataset = self._create_tokenized_dataset(df)

        skf = StratifiedKFold(n_splits=10, shuffle=True)
        splits = skf.split(np.zeros(dataset.num_rows), dataset['labels'])

        results = {}
        best_model = None
        best_f1 = -1
        for indx, (train_idxs, val_idxs) in enumerate(splits):
            train_data = dataset.select(train_idxs)
            val_data = dataset.select(val_idxs)
            fold_output_dir = os.path.join(output_dir, f"fold_{indx}")
            eval_metrics, model = self.train(
                train_data, val_data, classes_weights, fold_output_dir)
            for metric, value in eval_metrics.items():
                metric = metric.split('eval_')[-1]
                if metric == 'F1' and value > best_f1:
                    best_model = model
                try:
                    results[metric].append(value)
                except:
                    results[metric] = [value]
        best_model.save_pretrained(os.path.join(output_dir, "best_model"))
        return results

    def train(self, train_data, val_data, classes_weights, output_dir):
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=self.num_labels)

        trainer = self._create_trainer(CustomTrainer=WeightedLossTrainer, model=model,
                                       compute_metrics=BertClassifier._compute_metrics,
                                       train_data=train_data, val_data=val_data, classes_weights=classes_weights,
                                       output_dir=output_dir, num_train_epochs=5)

        trainer.train()
        for f in os.listdir(output_dir):
            shutil.rmtree(os.path.join(output_dir, f))
        trainer.save_model(output_dir=os.path.join(output_dir, "best_model"))
        eval_metrics = trainer.evaluate()
        return eval_metrics, model

    def evaluate(self, df, output_dir):
        # Load model before evaluating
        dataset = self._create_tokenized_dataset(df)

        trainer = self._create_trainer(CustomTrainer=WeightedLossTrainer, model=self.model,
                                       compute_metrics=BertClassifier._compute_metrics, val_data=dataset,
                                       output_dir=output_dir)
        eval_metrics = trainer.evaluate()
        results = {}
        for metric, value in eval_metrics.items():
            results[metric.split('eval_')[-1]] = value
        return results

    def predict(self, df):
        dataset = self._create_tokenized_dataset(df)
        trainer = Trainer(model=self.model, data_collator=self.data_collator)
        predictions = trainer.predict(dataset)
        return predictions.predictions, predictions.label_ids

    @staticmethod
    def _compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'Accuracy': acc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }


class MLBertClassifier(BertBase):
    def __init__(self, labels_names, stratified_label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_names = labels_names
        self.stratified_label = stratified_label

    def cross_validation(self, df, output_dir):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        classes_weights = 1 / \
                          (np.array(df['labels'].tolist()).sum(axis=0) / len(df['labels']))
        classes_weights /= np.linalg.norm(classes_weights)
        # print(classes_weights)
        # classes_weights *= (len(df['labels']) / np.array(df['labels'].tolist()).min())
        # print(classes_weights)
        # exit()
        classes_weights = torch.tensor(classes_weights).to(device)

        dataset = self._create_tokenized_dataset(df)

        skf = StratifiedKFold(n_splits=10, shuffle=True)
        splits = skf.split(np.zeros(dataset.num_rows),
                           df[self.stratified_label])

        results = {}
        best_model = None
        best_f1 = -1
        for indx, (train_idxs, val_idxs) in enumerate(splits):
            train_data = dataset.select(train_idxs)
            val_data = dataset.select(val_idxs)
            fold_output_dir = os.path.join(output_dir, f"fold_{indx}")
            eval_metrics, model = self.train(
                train_data, val_data, classes_weights, fold_output_dir)
            for metric, value in eval_metrics.items():
                metric = metric.split('eval_')[-1]
                if metric == 'F1' and value > best_f1:
                    best_model = model
                try:
                    results[metric].append(value)
                except:
                    results[metric] = [value]
        best_model.save_pretrained(os.path.join(output_dir, "best_model"))
        return results

    def train(self, train_data, val_data, classes_weights, output_dir):
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=self.num_labels,
                                                                   problem_type="multi_label_classification")
        trainer = self._create_trainer(CustomTrainer=MultiLabelTrainer, model=model,
                                       compute_metrics=self._compute_metrics,
                                       train_data=train_data, val_data=val_data, classes_weights=classes_weights,
                                       output_dir=output_dir, num_train_epochs=10)
        trainer.train()
        for f in os.listdir(output_dir):
            shutil.rmtree(os.path.join(output_dir, f))
        trainer.save_model(output_dir=os.path.join(output_dir, "best_model"))
        eval_metrics = trainer.evaluate()
        return eval_metrics, model

    def _compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions >= 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        metrics = {'Accuracy': acc,
                   'F1': f1,
                   'Precision': precision,
                   'Recall': recall
                   }
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds)
        for i, l in enumerate(self.labels_names):
            metrics[f'F1_{l}'] = f1[i]
            metrics[f'Precision_{l}'] = precision[i]
            metrics[f'Recall_{l}'] = recall[i]
        return metrics


def save_json(data, file_name, folder_path):  # DELETE LATER
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, file_name), 'w') as f:
        json.dump(data, f, indent=4)


def evaluate_fairness_by_proportionality_and_equality():
    bert_classifier = BertClassifier()
    corpus = 'All'
    df = pd.read_csv('Data/MFTC_final.csv', index_col=0)
    df_label = df[['text', 'Fairness']]
    df_label = df_label.rename(columns={'Fairness': "labels"})
    models_predictions = []
    for label in ["Equality", "Proportionality"]:
        model_path = os.path.join(os.getcwd(), "bert_mfrc_ckp", label, corpus, 'best_model')
        bert_classifier.load_model(model_path)
        predictions, labels = bert_classifier.predict(df_label)
        models_predictions.append(predictions.argmax(-1))

    union_predictions = models_predictions[0] | models_predictions[0]

    precision, recall, f1, _ = precision_recall_fscore_support(labels, union_predictions, average='binary')
    acc = accuracy_score(labels, union_predictions)
    results = {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }
    save_json(data={corpus: results},
              file_name="info.json",
              folder_path=os.path.join('Results', 'bert_mfrc_on_mftc', 'Fairness'))


def evaluate_cross_domain(dataset_name, model_name):
    if dataset_name == 'mftc':
        labels = ["Care", "Loyalty", "Authority", "Purity"]
    elif dataset_name == 'mfrc':
        df = pd.read_csv('Data/MFRC_final.csv', index_col=0)
        labels = ["Fairness", "Care", "Loyalty", "Authority", "Purity"]
        df['Fairness'] = df["Equality"] | df["Proportionality"]

    result_folder = 'bert_' + model_name + '_on_' + dataset_name
    corpus = 'All'
    bert_classifier = BertClassifier()
    for label in labels:
        if dataset_name == 'mftc':
            df = pd.read_csv(f'Data/MFTC_{label}.csv', index_col=0)
        df_label = df[['text', label]]
        df_label = df_label.rename(columns={label: "labels"})

        model_path = os.path.join(os.getcwd(), f"bert_{model_name}_ckp", label, corpus, 'best_model')
        output_dir = os.path.join(os.getcwd(), "bert_ckp", label, corpus, 'MFRC', 'evaluate')
        bert_classifier.load_model(model_path)
        results = bert_classifier.evaluate(df_label, output_dir)
        results['model_path'] = model_path
        print(results)

        save_json(data={corpus: results},
                  file_name="info.json",
                  folder_path=os.path.join('Results', result_folder, label))


if __name__ == '__main__':
    # df = pd.read_csv('Data/MFRC_final.csv')
    # corpus = 'All'
    # use_ml = False
    # if use_ml:
    #     output_dir = os.path.join(os.getcwd(), "ml_bert_ckp", corpus, 'MF')
    #     print("Checkpoints are in the", output_dir)
    #     labels = ['Care', 'Equality', 'Proportionality',
    #               'Loyalty', 'Authority', 'Purity']
    #     df['labels'] = df[labels].values.astype(float).tolist()
    #     df['labels_string'] = df[labels].astype(str).agg(''.join, axis=1)
    #     ml_bert_classifier = MLBertClassifier(
    #         labels_names=labels, stratified_label='is_moral', num_labels=len(labels))
    #     results = ml_bert_classifier.cross_validation(
    #         df=df, output_dir=output_dir)
    # else:
    #     label_names = 'Care'

    #     df_label = df[['text', label_names]]
    #     df_label = df_label.rename(columns={label_names: "labels"})

    #     output_dir = os.path.join(os.getcwd(), "bert_ckp", corpus, label_names)

    #     bert_classifier = FrozenBertClassifier()
    #     results = bert_classifier.cross_validation(
    #         df=df_label, output_dir=output_dir)
    # print(results)

    evaluate_cross_domain(model_name='mftc', dataset_name='mfrc')
    evaluate_cross_domain(model_name='mfrc', dataset_name='mftc')
    evaluate_fairness_by_proportionality_and_equality()
import os.path
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, Pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from zsl.utils import log


def main():
    log(f"Loading the MLSUM German Dataset...", 20)
    data = load_dataset("ag_news")

    log(f"Initialize Multi-lingual NLI Model...", 20)
    model_name = "microsoft/deberta-v3-base"

    log(f"Loading Pre-Trained NLI Model...", 20)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    log(f"Loading Pre-Trained NLI Tokenizer...", 20)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log(f"Initializing ZSL Classification Pipeline...", 20)
    zsl_pipeline = pipeline(task="zero-shot-classification", model=model, tokenizer=tokenizer)

    log(f"Retrieving candidates, shuffle and select for testing...", 20)
    candidates = data['train'].features['label'].names

    map_kwargs = {
        "p": zsl_pipeline,
        "candidates": candidates,
        "prompt": "The topic (label) of this article is {}"
    }

    log(f"Testing the model...", 20)
    data["test"] = data["test"].map(predict, batched=True, batch_size=16, fn_kwargs=map_kwargs)

    log(f"Saving MLSUM Model Results...", 20)
    data.save_to_disk("results/agnews")

    log(f"Evaluating Predictions...", 20)
    actual_labels = data["test"]["label"]
    predicted_labels = data["test"].features["label"].str2int(data['test']['predicted_label'])
    accuracy = accuracy_score(y_true=actual_labels, y_pred=predicted_labels)

    log(f"The accuracy of the zero-shot classifier is {accuracy:,.2%}", 20)

    predicted_scores = data["test"].map(format_row, remove_columns=data["test"].column_names)
    predicted_scores = predicted_scores["scores"]
    all_labels = np.sort(np.unique(data['train']['label']))
    top_k_accuracy = top_k_accuracy_score(y_true=actual_labels, y_score=predicted_scores, k=3, labels=all_labels)

    log(f"The accuracy of the zero-shot classifier @top 3 is {top_k_accuracy:,.2%}", 20)

    results = pd.DataFrame({
        "prompt": "The topic is {}",
        "accuracy": accuracy,
        "accuracy_top_k": top_k_accuracy,
    }, index=[0])

    prompts = [
        'The article is about {}',
        'The text is about {}',
        'the category of this article is {}',
        'Topic: {}',
    ]

    log(f"About to experiment with different prompts...", 20)

    for t in prompts:
        log(f"Evaluating prompt: \"{t}\"...", 20)
        data_tmp = data.copy()
        map_kwargs = {
            "p": zsl_pipeline,
            "candidates": candidates,
            "prompt": t
        }
        data_tmp["test"] = data_tmp["test"].map(predict, batched=True, batch_size=16, fn_kwargs=map_kwargs)

        actual_labels = data_tmp["test"]["label"]
        predicted_labels = data_tmp["test"].features["label"].str2int(data_tmp['test']['predicted_label'])
        accuracy = accuracy_score(y_true=actual_labels, y_pred=predicted_labels)

        log(f"The accuracy of the zero-shot classifier for prompt \"{t}\" is {accuracy:,.2%}", 20)

        predicted_scores = data_tmp["test"].map(format_row, remove_columns=data_tmp["test"].column_names)
        predicted_scores = predicted_scores["scores"]
        all_labels = np.sort(np.unique(data_tmp['train']['label']))
        top_k_accuracy = top_k_accuracy_score(y_true=actual_labels, y_score=predicted_scores, k=3, labels=all_labels)

        log(f"The accuracy of the zero-shot classifier @top 3 for prompt \"{t}\" is {top_k_accuracy:,.2%}", 20)

        results = pd.concat([pd.DataFrame({
            "prompt": t,
            "accuracy": accuracy,
            "accuracy_top_k": top_k_accuracy,
        }, index=[0]), results.loc[:]]).reset_index(drop=True)

    log(f"Saving results...", 20)

    save_to_path = "results"
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)

    results.to_csv(f"{save_to_path}/{datetime.now().strftime('%Y_%m_%d_%H_%M')}_results.csv", index=False)

    log(f"Model has been successfully Evaluated!", 10)


def predict(batch, p, candidates: list, k: int = 3, prompt: str = "The topic is {}"):
    prediction = p(
        sequences=batch["text"],
        candidate_labels=candidates,
        hypothesis_template=prompt,
        multi_label=False
    )

    max_scores = [np.max(i["scores"]) for i in prediction]
    predicted_labels = [i["labels"][np.argmax(i["scores"])] for i in prediction]
    predicted_scores = [get_top_k_labels(i) for i in prediction]

    batch["predicted"] = prediction
    batch["predicted_label"] = predicted_labels
    batch["predicted_scores"] = max_scores
    batch[f"predicted_top_{k}"] = predicted_scores

    return batch


def format_row(row):
    labels = row["predicted"]["labels"]
    scores = row["predicted"]["scores"]

    sorted_labels = np.argsort(labels)

    ordered_labels = [labels[i] for i in sorted_labels]
    ordered_scores = [scores[i] for i in sorted_labels]

    return {
        "labels": ordered_labels,
        "scores": ordered_scores
    }


def get_top_k_labels(res: [], k: int = 3):
    labels = res["labels"]
    scores = res["scores"]

    k_idx = np.argpartition(scores, -k)[-k:]
    k_labels = [labels[i] for i in k_idx]

    return k_labels

import os.path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, silhouette_score
from gensim.models import Word2Vec
import pandas as pd


from zsl.utils import log, load_csv


def main(seen_path: str, unseen_path: str, class_header: str = 'disease', confused: bool = True, verbose: bool = False):
    log(f"Starting Zero-shot Learning using a Embedding and Clustering Method...", verbose=verbose)

    X_seen, y_seen = load_csv(seen_path, class_header, verbose)
    X_unseen, y_unseen = load_csv(unseen_path, class_header, verbose)

    log(f"Encoding Labels...", verbose=verbose)
    label_encoder = LabelEncoder()
    y_seen_encoded = label_encoder.fit_transform(y_seen)

    log(f"Retrieving \"Sentences\"...", verbose=verbose)
    sentences = [list(row) for row in X_seen.values]

    log(f"Creating embedding...", verbose=verbose)
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    X_seen_embedded = [model.wv[list(row)].mean(axis=0) for row in sentences]
    X_unseen_embedded = [model.wv[list(row)].mean(axis=0) for row in X_unseen.values]

    log(f"Predicting Unseen Classes...", verbose=verbose)

    predictions = []

    for sample in X_unseen_embedded:
        distances = [(i, pd.Series(sample).subtract(X_seen_embedded[i]).pow(2).sum()) for i in range(len(X_seen_embedded))]
        closest_indexes = min(distances, key=lambda x: x[1])
        closest_class = y_seen_encoded[closest_indexes[0]]
        predictions.append(closest_class)

    predictions_labels = label_encoder.inverse_transform(predictions)

    if confused:
        log(f"Creating confusion matrix... (this doesn't make much sense for clustering, but helps us manually validate on very small datasets.", verbose=verbose)
        matrix = confusion_matrix(y_unseen, predictions_labels)
        log(f"Confusion Matrix: \n {matrix}", 30)

        log(f"Computing Silhouette Average...",
            verbose=verbose)
        silhouette_avg = silhouette_score(X_unseen_embedded, predictions_labels)
        log(f"Silhouette Average: \n {silhouette_avg}", 20)

    return predictions_labels


def _euclidean(data, target):
    distances = []

    for class_index in range(len(data)):
        distance = pd.Series(target).substract(data[class_index]).pow(2).sum()
        distances.append((class_index, distance))

    return distances


def _manhatten(data, target):
    distances = []

    for class_index in range(len(data)):
        distance = pd.Series(target).substract(data[class_index]).abs().sum()
        distances.append((class_index, distance))

    return distances


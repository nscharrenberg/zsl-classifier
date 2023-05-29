import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from zsl.utils import log, load_csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix


def main(seen_path: str, unseen_path: str, class_header: str = 'disease', confused: bool = False, verbose: bool = False):
    log(f"Starting Zero-shot Learning using a Large Language Model and Clustering Method...", verbose=verbose)

    seen_features, seen_classes = load_csv(seen_path, class_header, verbose)
    unseen_features, unseen_classes = load_csv(unseen_path, class_header, verbose)

    model = SentenceTransformer('bert-base-uncased')
    seen_embeddings = model.encode(seen_features.values.tolist())
    unseen_embeddings = model.encode(unseen_features.values.tolist())

    predictions = []

    for embedding in unseen_embeddings:
        score = cosine_similarity([embedding], seen_embeddings)[0]
        predicted_label = seen_classes.tolist()[np.argmax(score)]
        predictions.append(predicted_label)

    if confused:
        log(f"Creating confusion matrix... (this doesn't make much sense for clustering, but helps us manually validate on very small datasets.)", verbose=verbose)
        matrix = confusion_matrix(unseen_classes.tolist(), predictions)
        log(f"Confusion Matrix: \n {matrix}", 30)

        log(f"Computing Silhouette Average...",
            verbose=verbose)
        silhouette_avg = silhouette_score(unseen_embeddings, predictions)
        log(f"Silhouette Average: \n {silhouette_avg}", 20)

    return predictions


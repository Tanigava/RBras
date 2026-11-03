import numpy as np

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from sklearn.manifold import trustworthiness


def calculate_trustworthiness(
    X_original,
    X_reduced,
    n_neighbors=15
):
    """
    Calcula Trustworthiness da representação reduzida.
    """

    return trustworthiness(
        X_original,
        X_reduced,
        n_neighbors=n_neighbors
    )


def calculate_silhouette(X, labels):
    """
    Calcula Silhouette.

    Para HDBSCAN, remove os pontos classificados como ruído (-1).
    """

    labels = np.asarray(labels)

    mask = labels != -1

    X_valid = X[mask]
    labels_valid = labels[mask]

    unique_labels = np.unique(labels_valid)

    if len(unique_labels) < 2:
        return np.nan

    if len(unique_labels) >= len(labels_valid):
        return np.nan

    return silhouette_score(
        X_valid,
        labels_valid
    )


def calculate_ari(labels_true, labels_pred):

    return adjusted_rand_score(
        labels_true,
        labels_pred
    )


def calculate_nmi(labels_true, labels_pred):

    return normalized_mutual_info_score(
        labels_true,
        labels_pred
    )


def calculate_metrics(
    X_original,
    X_reduced,
    labels_pred,
    labels_true=None,
    n_neighbors=15
):
    """
    Calcula as métricas do experimento.
    """

    results = {}

    results["Trustworthiness"] = calculate_trustworthiness(
        X_original,
        X_reduced,
        n_neighbors
    )

    results["Silhouette"] = calculate_silhouette(
        X_reduced,
        labels_pred
    )

    if labels_true is not None:

        results["ARI"] = calculate_ari(
            labels_true,
            labels_pred
        )

        results["NMI"] = calculate_nmi(
            labels_true,
            labels_pred
        )

    return results
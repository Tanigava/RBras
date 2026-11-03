import numpy as np
import igraph as ig
import leidenalg

import hdbscan

from sklearn.neighbors import NearestNeighbors


def run_hdbscan(
    X,
    min_cluster_size=15,
    min_samples=5
):
    """
    Aplica HDBSCAN.
    """

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    return labels, model


def run_leiden(
    X,
    n_neighbors=15,
    resolution=0.1,
    seed=42,
    weight_mode="none"
):
    """
    Constrói um grafo KNN e aplica Leiden.

    weight_mode:
        none       -> grafo não ponderado
        distance   -> utiliza as distâncias originais
        similarity -> converte distâncias em similaridades
    """

    if weight_mode not in {"none", "distance", "similarity"}:
        raise ValueError(
            "weight_mode deve ser none, distance ou similarity."
        )

    n_samples = X.shape[0]

    if n_neighbors >= n_samples:
        raise ValueError(
            "n_neighbors deve ser menor que o número de amostras."
        )

    knn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="euclidean"
    )

    knn.fit(X)

    distances, indices = knn.kneighbors(X)

    edges = {}

    for i in range(n_samples):

        for j, distance in zip(
            indices[i][1:],
            distances[i][1:]
        ):

            j = int(j)

            if i == j:
                continue

            edge = tuple(sorted((i, j)))

            if edge not in edges:
                edges[edge] = float(distance)

            else:
                edges[edge] = min(
                    edges[edge],
                    float(distance)
                )

    edge_list = list(edges.keys())

    graph = ig.Graph(
        n=n_samples,
        edges=edge_list,
        directed=False
    )

    if weight_mode == "distance":

        weights = [
            edges[edge]
            for edge in edge_list
        ]

    elif weight_mode == "similarity":

        distances_values = np.array(
            [edges[edge] for edge in edge_list]
        )

        scale = np.median(distances_values)

        if scale <= 0:
            scale = 1.0

        weights = np.exp(
            -distances_values / scale
        ).tolist()

    else:
        weights = None

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed
    )

    labels = np.array(partition.membership)

    return labels, partition
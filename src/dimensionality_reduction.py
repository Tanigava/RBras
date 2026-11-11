import pandas as pd

from sklearn.decomposition import PCA

from pacmap import PaCMAP
from umap import UMAP


def run_pca(X, n_components=50, random_state=42):
    """
    Aplica PCA e retorna a representação reduzida.
    """

    model = PCA(
        n_components=n_components,
        random_state=random_state
    )

    embedding = model.fit_transform(X)

    return embedding, model



def run_pacmap(
    X,
    n_neighbors=10,
    MN_ratio=0.5,
    FP_ratio=2.0,
    n_components=2,
    random_state=42,
    verbose = False
):
    import pacmap

    model = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state,
        verbose = False
    )

    embedding = model.fit_transform(X)

    return embedding, model


def run_umap(
    X,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
    verbose = False
):
    """
    Aplica UMAP.
    """

    model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        verbose = False
    )

    embedding = model.fit_transform(X)

    return embedding, model


def save_embedding(
    embedding,
    sample_ids,
    output_path,
    prefix="Dim"
):
    """
    Salva a representação reduzida preservando
    os identificadores das amostras.
    """

    columns = [
        f"{prefix}_{i+1}"
        for i in range(embedding.shape[1])
    ]

    df = pd.DataFrame(
        embedding,
        index=sample_ids,
        columns=columns
    )

    df.to_csv(output_path)

    return df
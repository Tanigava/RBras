
import itertools
import json
import io
import warnings
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.dimensionality_reduction import (
    run_pacmap,
    run_umap
)

from src.clustering import (
    run_hdbscan,
    run_leiden
)

from src.metrics import calculate_metrics


@contextmanager
def _model_output_context(verbose=True):
    """Controla a saída textual e os warnings emitidos pelos modelos.

    Quando verbose=False, suprime prints/stdout e warnings convencionais
    durante a execução do modelo. Exceções continuam sendo propagadas.
    """
    if verbose:
        yield
    else:
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield


# ============================================================
# 1. GERAR COMBINAÇÕES DE PARÂMETROS
# ============================================================

def generate_combinations(param_grid):
    """
    Gera todas as combinações possíveis de parâmetros.
    """

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = itertools.product(*values)

    return [
        dict(zip(keys, combination))
        for combination in combinations
    ]


# ============================================================
# 2. EXECUTAR REDUÇÃO DE DIMENSIONALIDADE
# ============================================================

def run_reduction(
    X,
    method,
    params,
    random_state=42,
    verbose=True
):
    """
    Executa a técnica de redução escolhida.
    """

    params = params.copy()

    if method == "PaCMAP":

        if "n_neighbors" in params:
            params["n_neighbors"] = int(params["n_neighbors"])

        if "MN_ratio" in params:
            params["MN_ratio"] = float(params["MN_ratio"])

        if "FP_ratio" in params:
            params["FP_ratio"] = float(params["FP_ratio"])

        with _model_output_context(verbose):
            embedding, model = run_pacmap(
                X,
                **params,
                random_state=random_state
            )

    elif method == "UMAP":

        if "n_neighbors" in params:
            params["n_neighbors"] = int(params["n_neighbors"])

        if "min_dist" in params:
            params["min_dist"] = float(params["min_dist"])

        with _model_output_context(verbose):
            embedding, model = run_umap(
                X,
                **params,
                random_state=random_state
            )

    elif method == "PCA":

        embedding = X
        model = None

    else:

        raise ValueError(
            f"Método de redução desconhecido: {method}"
        )

    return embedding, model


# ============================================================
# 3. EXECUTAR CLUSTERING
# ============================================================

def run_clustering(
    X,
    method,
    params,
    random_state=42,
    verbose=True
):
    """
    Executa o algoritmo de agrupamento escolhido.
    """

    params = params.copy()

    if method == "HDBSCAN":

        if "min_cluster_size" in params:
            params["min_cluster_size"] = int(
                params["min_cluster_size"]
            )

        if "min_samples" in params:
            params["min_samples"] = int(
                params["min_samples"]
            )

        with _model_output_context(verbose):
            labels, model = run_hdbscan(
                X,
                **params
            )

    elif method == "Leiden":

        if "n_neighbors" in params:
            params["n_neighbors"] = int(
                params["n_neighbors"]
            )

        if "resolution" in params:
            params["resolution"] = float(
                params["resolution"]
            )

        with _model_output_context(verbose):
            labels, model = run_leiden(
                X,
                seed=random_state,
                **params
            )

    else:

        raise ValueError(
            f"Método de clustering desconhecido: {method}"
        )

    return labels, model


# ============================================================
# 4. GRID SEARCH
# ============================================================

def run_grid_search(
    X,
    labels_true,
    reduction_grids,
    clustering_grids,
    random_state=42,
    n_neighbors_metric=15,
    verbose=True,
    show_errors=True
):
    """
    Executa o Grid Search combinando técnicas de redução
    de dimensionalidade e algoritmos de clustering.

    Para cada combinação de parâmetros de redução, a projeção
    é calculada uma única vez e reutilizada nos algoritmos
    de clustering.
    """

    results = []

    for reduction_method, reduction_grid in reduction_grids.items():

        reduction_combinations = generate_combinations(
            reduction_grid
        )

        for reduction_params in reduction_combinations:

            if verbose:
                print(
                    f"Redução: {reduction_method} | "
                    f"Parâmetros: {reduction_params}"
                )

            try:

                embedding, _ = run_reduction(
                    X,
                    reduction_method,
                    reduction_params,
                    random_state=random_state,
                    verbose=verbose
                )

            except Exception as error:

                if show_errors:
                    print(
                        f"Erro na redução {reduction_method}: {error}"
                    )

                continue

            for clustering_method, clustering_grid in clustering_grids.items():

                clustering_combinations = generate_combinations(
                    clustering_grid
                )

                for clustering_params in clustering_combinations:

                    if verbose:
                        print(
                            f"Clustering: {clustering_method} | "
                            f"Parâmetros: {clustering_params}"
                        )

                    try:

                        labels_pred, _ = run_clustering(
                            embedding,
                            clustering_method,
                            clustering_params,
                            random_state=random_state,
                            verbose=verbose
                        )

                        metrics = calculate_metrics(
                            X,
                            embedding,
                            labels_pred,
                            labels_true,
                            n_neighbors_metric
                        )

                        labels_pred = np.asarray(labels_pred)

                        row = {
                            "Reduction": reduction_method,
                            "Clustering": clustering_method,

                            **{
                                f"Reduction_{key}": value
                                for key, value in reduction_params.items()
                            },

                            **{
                                f"Clustering_{key}": value
                                for key, value in clustering_params.items()
                            },

                            "N_Clusters": len(
                                set(labels_pred) - {-1}
                            ),

                            "N_Noise": int(
                                np.sum(labels_pred == -1)
                            ),

                            **metrics
                        }

                        results.append(row)

                    except Exception as error:

                        if show_errors:
                            print(
                                f"Erro em {clustering_method}: {error}"
                            )

                        continue

    return pd.DataFrame(results)


# ============================================================
# 5. CONVERTER TIPOS NUMPY PARA JSON
# ============================================================

def convert_numpy_types(obj):
    """
    Converte tipos NumPy em tipos nativos do Python,
    permitindo a serialização em JSON.
    """

    if isinstance(obj, dict):

        return {
            key: convert_numpy_types(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):

        return [
            convert_numpy_types(value)
            for value in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


# ============================================================
# 6. SELECIONAR OS MELHORES MODELOS
# ============================================================

def select_best_models(
    results,
    metric="Silhouette",
    output_dir=None,
    group_by=("Cenario", "Reduction", "Clustering")
):
    """
    Seleciona o melhor modelo de cada combinação de cenário,
    redução e clustering.

    O critério padrão é a maior Silhouette.

    Salva:
        - best_models.csv
        - best_models_params.json
    """

    if results.empty:

        raise ValueError(
            "O DataFrame de resultados está vazio."
        )

    if metric not in results.columns:

        raise ValueError(
            f"A métrica '{metric}' não foi encontrada."
        )

    missing_columns = [
        col for col in group_by
        if col not in results.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Colunas não encontradas: {missing_columns}"
        )

    # Garantir que a métrica seja numérica
    valid_results = results.copy()

    valid_results[metric] = pd.to_numeric(
        valid_results[metric],
        errors="coerce"
    )

    # Remover resultados sem métrica válida
    valid_results = valid_results.dropna(
        subset=[metric]
    )

    if valid_results.empty:

        raise ValueError(
            f"Não existem resultados válidos para '{metric}'."
        )

    # Ordenar do maior para o menor
    valid_results = valid_results.sort_values(
        by=metric,
        ascending=False
    )

    # Selecionar o melhor modelo de cada grupo
    best_models = (
        valid_results
        .groupby(
            list(group_by),
            dropna=False,
            sort=False
        )
        .head(1)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Extrair parâmetros específicos de cada método
    # --------------------------------------------------------

    parameters = []

    for _, row in best_models.iterrows():

        # Parâmetros de redução
        if row["Reduction"] == "PaCMAP":

            reduction_params = {
                "n_neighbors": int(row["Reduction_n_neighbors"]),
                "MN_ratio": float(row["Reduction_MN_ratio"]),
                "FP_ratio": float(row["Reduction_FP_ratio"])
            }

        elif row["Reduction"] == "UMAP":

            reduction_params = {
                "n_neighbors": int(row["Reduction_n_neighbors"]),
                "min_dist": float(row["Reduction_min_dist"])
            }

        else:

            raise ValueError(
                f"Método de redução não reconhecido: {row['Reduction']}"
            )

        # Parâmetros de clustering
        if row["Clustering"] == "HDBSCAN":

            clustering_params = {
                "min_cluster_size": int(
                    row["Clustering_min_cluster_size"]
                ),
                "min_samples": int(
                    row["Clustering_min_samples"]
                )
            }

        elif row["Clustering"] == "Leiden":

            clustering_params = {
                "n_neighbors": int(
                    row["Clustering_n_neighbors"]
                ),
                "resolution": float(
                    row["Clustering_resolution"]
                )
            }

        else:

            raise ValueError(
                f"Método de clustering não reconhecido: {row['Clustering']}"
            )

        # Verificar ausência de parâmetros
        if any(
            pd.isna(value)
            for value in reduction_params.values()
        ):

            raise ValueError(
                f"Parâmetros ausentes na redução: {row['Reduction']}"
            )

        if any(
            pd.isna(value)
            for value in clustering_params.values()
        ):

            raise ValueError(
                f"Parâmetros ausentes no clustering: {row['Clustering']}"
            )

        # Organizar informações do modelo
        entry = {
            "Cenario": row["Cenario"],
            "Reduction": row["Reduction"],
            "Clustering": row["Clustering"],
            "Metric": metric,
            "Score": float(row[metric]),
            "Reduction_params": reduction_params,
            "Clustering_params": clustering_params
        }

        parameters.append(entry)

    # Converter tipos NumPy
    parameters = convert_numpy_types(parameters)

    # --------------------------------------------------------
    # Salvar resultados
    # --------------------------------------------------------

    if output_dir is not None:

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Salvar tabela dos melhores modelos
        csv_path = output_dir / "best_models.csv"

        best_models.to_csv(
            csv_path,
            index=False
        )

        # Salvar parâmetros em JSON
        json_path = output_dir / "best_models_params.json"

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                parameters,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

        print(
            f"Resultados salvos em: {output_dir}"
        )

    return best_models

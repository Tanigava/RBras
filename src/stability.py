import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.experiments import run_reduction, run_clustering
from src.metrics import calculate_metrics


def load_best_models(path):
    """
    Carrega os parâmetros selecionados no Notebook 02.
    """
    with open(path, "r", encoding="utf-8") as file:
        best_models = json.load(file)

    return best_models


def calculate_partition_stability(labels_by_seed):
    """
    Calcula a estabilidade das partições por meio do ARI
    entre todos os pares de execuções.

    labels_by_seed: lista de vetores de rótulos.
    """

    ari_values = []

    for labels_a, labels_b in combinations(labels_by_seed, 2):
        ari = adjusted_rand_score(labels_a, labels_b)
        ari_values.append(ari)

    return {
        "ARI_estabilidade_media": np.mean(ari_values),
        "ARI_estabilidade_std": np.std(ari_values, ddof=1)
        if len(ari_values) > 1 else 0.0
    }


def run_stability_analysis(
    X,
    best_models,
    labels_true=None,
    n_runs=10,
    base_seed=42,
    n_neighbors_metric=15,
    output_dir=None,
    verbose=True
):
    """
    Executa testes de estabilidade para os modelos selecionados.

    Os parâmetros dos modelos são mantidos fixos. Apenas a
    semente aleatória é alterada entre as execuções.

    Retorna:
        df_runs: métricas de cada execução.
        df_summary: resumo das métricas e da estabilidade.
        labels_by_model: rótulos produzidos em cada execução.
    """

    X = np.asarray(X, dtype=float)

    if labels_true is not None:
        labels_true = np.asarray(labels_true)

    results = []
    labels_by_model = {}

    for model_number, model_info in enumerate(best_models):

        scenario = model_info["Cenario"]
        reduction_method = model_info["Reduction"]
        clustering_method = model_info["Clustering"]

        reduction_params = model_info["Reduction_params"]
        clustering_params = model_info["Clustering_params"]

        model_id = (
            f"{scenario}_{reduction_method}_{clustering_method}"
        )

        if verbose:
            print(f"\nModelo: {model_id}")

        labels_by_seed = []

        # Executar o modelo com diferentes sementes
        for run in range(n_runs):

            seed = base_seed + run

            if verbose:
                print(f"Execução {run + 1}/{n_runs} | Seed: {seed}")

            # Redução de dimensionalidade
            embedding, _ = run_reduction(
                X,
                reduction_method,
                reduction_params,
                random_state=seed
            )

            # Clustering
            labels_pred, _ = run_clustering(
                embedding,
                clustering_method,
                clustering_params,
                random_state=seed
            )

            labels_pred = np.asarray(labels_pred)
            labels_by_seed.append(labels_pred)

            # Métricas de avaliação
            try:
                metrics = calculate_metrics(
                    X,
                    embedding,
                    labels_pred,
                    labels_true,
                    n_neighbors_metric
                )
            except Exception as error:
                if verbose:
                    print(f"Erro no cálculo das métricas: {error}")
                metrics = {}

            row = {
                "Model_ID": model_id,
                "Cenario": scenario,
                "Reduction": reduction_method,
                "Clustering": clustering_method,
                "Run": run + 1,
                "Seed": seed,
                "N_Clusters": len(set(labels_pred) - {-1}),
                "N_Noise": int(np.sum(labels_pred == -1)),
                **metrics
            }

            results.append(row)

        # Guardar os rótulos de cada execução
        labels_by_model[model_id] = pd.DataFrame(
            np.column_stack(labels_by_seed),
            index=np.arange(X.shape[0]),
            columns=[
                f"Seed_{base_seed + i}"
                for i in range(n_runs)
            ]
        )

    # Organizar resultados individuais
    df_runs = pd.DataFrame(results)

    # Calcular estabilidade e resumir métricas
    summary = []

    metric_columns = [
        col for col in [
            "Trustworthiness",
            "Silhouette",
            "ARI",
            "NMI",
            "N_Clusters",
            "N_Noise"
        ]
        if col in df_runs.columns
    ]

    for model_id, group in df_runs.groupby("Model_ID"):

        model_summary = {
            "Model_ID": model_id,
            "Cenario": group["Cenario"].iloc[0],
            "Reduction": group["Reduction"].iloc[0],
            "Clustering": group["Clustering"].iloc[0],
            "N_Runs": len(group)
        }

        # Média e desvio-padrão das métricas
        for metric in metric_columns:
            model_summary[f"{metric}_media"] = group[metric].mean()
            model_summary[f"{metric}_std"] = group[metric].std()

        # Estabilidade entre as partições
        stability = calculate_partition_stability(
            labels_by_model[model_id].to_numpy().T
        )

        model_summary.update(stability)

        summary.append(model_summary)

    df_summary = pd.DataFrame(summary)

    # Salvar resultados
    if output_dir is not None:

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_runs.to_csv(
            output_dir / "stability_runs.csv",
            index=False
        )

        df_summary.to_csv(
            output_dir / "stability_summary.csv",
            index=False
        )

        for model_id, labels in labels_by_model.items():
            labels.to_csv(
                output_dir / f"labels_{model_id}.csv"
            )

        if verbose:
            print(f"\nResultados salvos em: {output_dir}")

    return df_runs, df_summary, labels_by_model
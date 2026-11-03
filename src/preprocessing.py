from pathlib import Path
import pandas as pd



def load_clinical_data(path):
    """
    Carrega a tabela clínica e constrói um dicionário
    relacionando o identificador do paciente ao subtipo.
    """

    pacientes = pd.read_csv(
        path,
        sep=",",
        index_col="PATIENT_ID"
    )

    pacientes.index = (
        pacientes.index.astype(str)
        .str.replace(".", "-", regex=False)
    )

    pacientes["SUBTYPE"] = (
        pacientes["SUBTYPE"].fillna("Desconhecido")
    )

    dict_subtipos = {
        str(idx)[:12]: row["SUBTYPE"]
        for idx, row in pacientes.iterrows()
    }

    return pacientes, dict_subtipos


def load_expression_matrix(path):
    """
    Carrega uma matriz de expressão gênica,
    identifica sua orientação e remove possíveis
    colunas clínicas.
    """

    df = pd.read_csv(path, index_col=0)

    df.index = (
        df.index.astype(str)
        .str.replace(".", "-", regex=False)
    )

    # Identificar orientação amostras x genes
    if df.shape[0] > df.shape[1]:
        df = df.T

    # Remover possíveis colunas clínicas
    colunas_clinicas = [
        "PAM50",
        "SUBTYPE",
        "CLASS",
        "ROTULO",
        "BRCA_SUBTYPE"
    ]

    df = df.drop(
        columns=[
            col for col in df.columns
            if str(col).upper() in colunas_clinicas
        ],
        errors="ignore"
    )

    return df


def load_datasets(mad_dir):
    """
    Carrega os quatro cenários de expressão gênica
    armazenados na pasta mad_datasets.
    """

    cenarios_dados = {
        "MAD 3.000": "VST_ZSCORE_MAD_3000.csv",
        "MAD 5.000": "VST_ZSCORE_MAD_5000.csv",
        "MAD 10.000": "VST_ZSCORE_MAD_10000.csv",
        "MAD 20.000": "VST_ZSCORE_MAD_20000.csv"
    }

    datasets = {}

    for nome, arquivo in cenarios_dados.items():

        caminho = Path(mad_dir) / arquivo

        if not caminho.exists():
            print(f"Arquivo não encontrado: {caminho}")
            continue

        df = load_expression_matrix(caminho)

        datasets[nome] = df

        print(f"\n{nome}")
        print("Dimensões:", df.shape)

    return datasets


def map_patient_subtypes(sample_ids, dict_subtipos):
    """
    Associa cada amostra ao subtipo clínico PAM50,
    utilizando os primeiros 12 caracteres do identificador.
    """

    return pd.Series(
        [
            dict_subtipos.get(
                str(idx)[:12],
                "Desconhecido"
            )
            for idx in sample_ids
        ],
        index=sample_ids,
        name="SUBTYPE"
    )

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

ARQUIVO_CSV = "Tabelas/resultados.csv"


def treinar_modelo():

    df = pd.read_csv(ARQUIVO_CSV)

    X = df[
        [
            "batalha",
            "efeito_buff",
            "personagem_1",
            "personagem_2",
            "personagem_3",
            "personagem_4",
        ]
    ]

    y = df["pontuacao_media"]

    categorias = [
        "batalha",
        "efeito_buff",
        "personagem_1",
        "personagem_2",
        "personagem_3",
        "personagem_4",
    ]

    preprocessador = ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(handle_unknown="ignore"),
                categorias,
            )
        ]
    )

    modelo = RandomForestRegressor(
        n_estimators = 300,
        random_state = 42
    )

    pipeline = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", modelo),
    ])

    pipeline.fit(X, y)

    return pipeline


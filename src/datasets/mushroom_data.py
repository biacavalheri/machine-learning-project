
import pandas as pd
from ucimlrepo import fetch_ucirepo


def carregar_mushroom():
    """Carrega o dataset Mushroom da UCI."""

    mushroom = fetch_ucirepo(id=73)

    X = mushroom.data.features
    y = mushroom.data.targets.iloc[:, 0]

    return X, y


def explorar_mushroom(X, y):
    """Apresenta informações básicas sobre o dataset."""

    df = pd.concat([X, y], axis=1)

    print("\n=== MUSHROOM ===")
    print(f"Observações: {df.shape[0]}")
    print(f"Atributos preditores: {X.shape[1]}")

    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nValores ausentes:")
    print(df.isnull().sum())

    print("\nDistribuição das classes:")
    print(y.value_counts())

    print("\nDistribuição percentual das classes:")
    print((y.value_counts(normalize=True) * 100).round(2))

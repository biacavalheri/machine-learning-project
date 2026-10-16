
import pandas as pd
from ucimlrepo import fetch_ucirepo


def carregar_concrete():
    """Carrega o dataset Concrete Compressive Strength da UCI."""

    concrete = fetch_ucirepo(id=165)

    X = concrete.data.features
    y = concrete.data.targets.iloc[:, 0]

    return X, y


def explorar_concrete(X, y):
    """Apresenta informações básicas sobre o dataset."""

    df = pd.concat([X, y], axis=1)

    print("\n=== CONCRETE COMPRESSIVE STRENGTH ===")
    print(f"Observações: {df.shape[0]}")
    print(f"Atributos preditores: {X.shape[1]}")

    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nValores ausentes:")
    print(df.isnull().sum())

    print("\nEstatísticas descritivas:")
    print(df.describe().round(2))


import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def dividir_concrete(X, y):
    """Divide os dados sem compartilhar composições idênticas."""

    # Cada combinação única de atributos recebe um grupo
    grupos = pd.factorize(
        pd.MultiIndex.from_frame(X)
    )[0]

    divisao = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    indices_treino, indices_teste = next(
        divisao.split(X, y, groups=grupos)
    )

    X_train = X.iloc[indices_treino]
    X_test = X.iloc[indices_teste]

    y_train = y.iloc[indices_treino]
    y_test = y.iloc[indices_teste]

    print("\n=== DIVISÃO DOS DADOS ===")
    print(f"Treinamento: {len(X_train)} observações")
    print(f"Teste: {len(X_test)} observações")

    grupos_train = grupos[indices_treino]

    return X_train, X_test, y_train, y_test, grupos_train
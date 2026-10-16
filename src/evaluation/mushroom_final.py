
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def avaliar_modelos_finais(modelos, X_train, X_test, y_train, y_test):
    """Treina e avalia os classificadores no conjunto de teste."""

    Path("results").mkdir(exist_ok=True)

    resultados = []
    matrizes = {}

    for nome, modelo in modelos.items():
        modelo_treinado = clone(modelo)
        modelo_treinado.fit(X_train, y_train)

        y_pred = modelo_treinado.predict(X_test)

        matriz = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        matrizes[nome] = matriz

        resultados.append({
            "Modelo": nome,
            "Acurácia": accuracy_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "Falsos positivos": matriz[0, 1],
            "Falsos negativos": matriz[1, 0]
        })

    tabela = pd.DataFrame(resultados)
    tabela = tabela.sort_values(
        ["Recall", "F1"],
        ascending=False
    )


    return tabela, matrizes

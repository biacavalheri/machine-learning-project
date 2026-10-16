
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate


def avaliar_classificacao_cv(modelo, X_train, y_train):
    """Avalia um classificador por validação cruzada estratificada."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    metricas = {
        "Acurácia": "accuracy",
        "Precisão": "precision",
        "Recall": "recall",
        "F1": "f1"
    }

    scores = cross_validate(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=metricas,
        n_jobs=-1
    )

    return pd.DataFrame({
        "Métrica": list(metricas.keys()),
        "Média": [
            scores[f"test_{nome}"].mean()
            for nome in metricas
        ],
        "Desvio-padrão": [
            scores[f"test_{nome}"].std()
            for nome in metricas
        ]
    })

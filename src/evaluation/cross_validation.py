
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_validate


def comparar_modelos_cv(modelos, X_train, y_train, grupos_train):
    """Compara modelos por validação cruzada com grupos."""

    cv = GroupKFold(n_splits=5)

    metricas = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    resultados = []

    for nome, modelo in modelos.items():
        scores = cross_validate(
            clone(modelo),
            X_train,
            y_train,
            groups=grupos_train,
            cv=cv,
            scoring=metricas,
            n_jobs=-1
        )

        resultados.append({
            "Modelo": nome,
            "MAE": -np.mean(scores["test_MAE"]),
            "RMSE": -np.mean(scores["test_RMSE"]),
            "R²": np.mean(scores["test_R2"]),
            "Desvio RMSE": np.std(scores["test_RMSE"])
        })

    return pd.DataFrame(resultados).sort_values("RMSE")

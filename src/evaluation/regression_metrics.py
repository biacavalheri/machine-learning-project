
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def avaliar_regressao(modelo, X_test, y_test):
    """Avalia um modelo de regressão."""

    y_pred = modelo.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    }

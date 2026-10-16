
from sklearn.ensemble import RandomForestRegressor


def treinar_random_forest(X_train, y_train):
    """Treina um Random Forest para regressão."""

    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    return modelo

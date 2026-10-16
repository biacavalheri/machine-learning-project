
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor


def treinar_bagging(X_train, y_train):
    """Treina um ensemble Bagging com árvores de decisão."""

    arvore_base = DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=2,
        random_state=42
    )

    modelo = BaggingRegressor(
        estimator=arvore_base,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    return modelo

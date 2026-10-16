
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def treinar_adaboost(X_train, y_train):
    """Treina um AdaBoost para regressão."""

    arvore_base = DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    )

    modelo = AdaBoostRegressor(
        estimator=arvore_base,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    return modelo

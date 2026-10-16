
from sklearn.tree import DecisionTreeRegressor


def treinar_arvore_decisao(X_train, y_train):
    """Treina uma árvore de decisão para regressão."""

    modelo = DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    return modelo

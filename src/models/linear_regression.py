
from sklearn.linear_model import LinearRegression


def treinar_regressao_linear(X_train, y_train):
    """Treina um modelo de regressão linear."""

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    return modelo

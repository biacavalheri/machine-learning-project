
from sklearn.model_selection import train_test_split


def dividir_mushroom(X, y):
    """Divide os dados preservando a proporção das classes."""

    y = y.map({"e": 0, "p": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n=== DIVISÃO DOS DADOS — MUSHROOM ===")
    print(f"Treinamento: {len(X_train)} observações")
    print(f"Teste: {len(X_test)} observações")

    print("\nDistribuição das classes no treinamento:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nDistribuição das classes no teste:")
    print(y_test.value_counts(normalize=True).round(4))

    return X_train, X_test, y_train, y_test

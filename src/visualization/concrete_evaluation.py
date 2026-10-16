
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def grafico_reais_previstos(modelos, X_test, y_test):
    """Compara valores reais e previstos pelos modelos."""

    RESULTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, len(modelos), figsize=(12, 5))

    minimo = y_test.min()
    maximo = y_test.max()

    for ax, (nome, modelo) in zip(axes, modelos.items()):
        previsoes = modelo.predict(X_test)

        ax.scatter(y_test, previsoes, alpha=0.5)
        ax.plot(
            [minimo, maximo],
            [minimo, maximo],
            "r--",
            label="Previsão perfeita"
        )

        ax.set_title(nome)
        ax.set_xlabel("Resistência real (MPa)")
        ax.set_ylabel("Resistência prevista (MPa)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "concrete_reais_vs_previstos.png",
        dpi=300
    )
    plt.close()

    print("\nGráfico de avaliação salvo em results/")

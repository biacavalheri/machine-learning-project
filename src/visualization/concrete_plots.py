
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def gerar_graficos_concrete(X, y):
    """Gera os gráficos exploratórios do Concrete."""

    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.concat([X, y], axis=1)

    # Matriz de correlação
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Matriz de Correlação — Concrete")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "concrete_correlacao.png", dpi=300)
    plt.close()

    # Dispersão dos principais atributos
    atributos = ["Cement", "Superplasticizer", "Age", "Water"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, atributo in zip(axes.flat, atributos):
        sns.scatterplot(
            x=X[atributo],
            y=y,
            ax=ax,
            alpha=0.5
        )
        ax.set_title(f"{atributo} × Resistência")
        ax.set_ylabel("Resistência (MPa)")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "concrete_dispersao.png", dpi=300)
    plt.close()

    print("\nGráficos exploratórios salvos em results/")

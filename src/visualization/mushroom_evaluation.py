
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plotar_matrizes_confusao(matrizes):
    """Gera as matrizes de confusão dos classificadores."""

    Path("results").mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))

    for ax, (nome, matriz) in zip(axes.flat, matrizes.items()):
        display = ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=["Comestível", "Venenoso"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )

        ax.set_title(nome)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")

    plt.suptitle(
        "Matrizes de Confusão — Mushroom",
        fontsize=16
    )
    plt.tight_layout()

    plt.savefig(
        "results/mushroom_matrizes_confusao.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

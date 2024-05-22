import os
import sys

sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from experiments.utils import (
    set_plt,
    MODELS_ORDER,
)

from experiments.features_ablation import ATTACKS_FTS_INDICES, FTS_PRETTY_NAMES
from experiments.trainset_size_ablation import SIZES_TO_SHOW, X_MEM_SIZE


set_plt()

PATH_TO_PLOTS = "experiments/out/ablation"


def get_ablation_heatmap_fts(df: pd.DataFrame, ax: plt.Axes):
    df = df.groupby(["Attack", "Model", "n"]).pvalue.mean().reset_index()
    df = df.loc[df.pvalue <= 0.01].groupby(["Attack", "Model"]).n.min().reset_index()
    df = df.pivot(index="Attack", columns="Model", values="n").loc[
        [k for k in ATTACKS_FTS_INDICES.keys()]
    ][MODELS_ORDER]
    df = df.rename(index=FTS_PRETTY_NAMES).T
    g = sns.heatmap(
        df,
        cmap="RdYlGn_r",
        norm=LogNorm(),
        annot=True,
        fmt=".5g",
        mask=df.isna(),
        ax=ax,
        cbar=False,
    )
    g.set_facecolor("black")
    ax.collections[0].set_clim(30, 20_000)
    ax.set(
        xlabel="",
        ylabel="",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.vlines([4, 8, 14, 18], *ax.get_ylim(), color="blue", lw=3)


def get_ablation_heatmap(df: pd.DataFrame, ax: plt.Axes):
    df = df.groupby([X_MEM_SIZE, "Model", "n"]).pvalue.mean().reset_index()
    df = df.loc[df.pvalue <= 0.01].groupby([X_MEM_SIZE, "Model"]).n.min().reset_index()
    df = df.pivot(index=X_MEM_SIZE, columns="Model", values="n")
    df = df.loc[[size for size in SIZES_TO_SHOW if size in df.index]][MODELS_ORDER].T
    g = sns.heatmap(
        df,
        cmap="RdYlGn_r",
        norm=LogNorm(),
        annot=True,
        fmt=".5g",
        mask=df.isna(),
        ax=ax,
        cbar=True,
        cbar_kws={
            "label": "Number of samples in $\mathbf{Q}_{sus}$",
            "aspect": 10,
        },
    )
    g.set_facecolor("black")
    ax.collections[0].set_clim(30, 20_000)
    ax.set(
        xlabel=X_MEM_SIZE,
        ylabel="",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels([])
    ax.set_yticks([])


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    df_features = pd.read_csv(
        f"experiments/out/features_ablation/pvalue_per_sample.csv",
    )
    df_train = pd.read_csv(
        f"experiments/out/trainset_size_ablation/pvalue_per_sample.csv",
    )
    fig = plt.figure(figsize=(30, 8))
    ax_features = fig.add_subplot(1, 4, (1, 3))
    ax_train = fig.add_subplot(1, 4, 4)
    fig.subplots_adjust(wspace=0.05)

    get_ablation_heatmap_fts(df_features, ax_features)
    get_ablation_heatmap(df_train, ax_train)

    plt.savefig(
        f"{PATH_TO_PLOTS}/ablation_heatmaps.pdf", format="pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()

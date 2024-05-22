import os
import sys

sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product

from src import get_p_value
from experiments.utils import (
    set_plt,
    ATTACKS_NAME_MAPPING,
    MODELS_NAME_MAPPING,
    MODELS_COLORS,
    MODELS_MARKERS,
    MODELS_ORDER,
    RESAMPLING_CNT,
    MODELS,
    RUN_ID,
    OURS,
)
import pandas as pd

set_plt()

NSAMPLES_TO_SHOW = [100, 500, 1000, 5000]

PATH_TO_SCORES = "out/scores"
PATH_TO_PLOTS = "experiments/out/pvalue_per_sample"


def plot_pvalues_cmp(
    title: str,
    df: pd.DataFrame,
    ax: plt.Axes,
    hue: str,
    xlabel: str = "Number of samples",
):
    sns.lineplot(
        data=df,
        x="n",
        y="pvalue",
        hue=hue,
        # style=hue,
        ax=ax,
        palette=(
            [MODELS_COLORS[model] for model in MODELS_ORDER] if hue == "Model" else None
        ),
        # markers=MODELS_MARKERS,
    )
    ax.plot(df.n, 0.05 * np.ones_like(df.n), "--", color="black", label="p-value: 0.05")
    ax.plot(df.n, 0.01 * np.ones_like(df.n), "--", color="green", label="p-value: 0.01")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=[10 ** (-3), 1],
        title=title,
        ylabel="p-value",
        xlabel=xlabel,
    )
    ax.get_legend().remove()


def get_data() -> list:
    out = []
    for attack, model in tqdm(product(["cdi"], MODELS)):
        data = np.load(
            f"{PATH_TO_SCORES}/{model}_{attack}_{RUN_ID}.npz", allow_pickle=True
        )
        members_lower = data["metadata"][()]["members_lower"]
        n_eval_samples = data["metadata"][()]["n_samples_eval"]
        members = data["members"][:n_eval_samples]
        nonmembers = data["nonmembers"][:n_eval_samples]

        nsamples = np.array(
            [n for n in range(2, 11)]
            + [n for n in range(20, 110, 10)]
            + [n for n in range(200, 1100, 100)]
            + [n for n in range(2000, 11000, 1000)]
            + [20000]
        )
        nsamples = nsamples[nsamples <= n_eval_samples]
        indices_total = np.arange(n_eval_samples)
        for n in nsamples:
            for r in range(RESAMPLING_CNT):
                indices = np.random.choice(indices_total, n, replace=False)
                pvalue, is_correct_order = get_p_value(
                    members[indices], nonmembers[indices], members_lower=members_lower
                )
                out.append(
                    [
                        ATTACKS_NAME_MAPPING[attack],
                        MODELS_NAME_MAPPING[model],
                        n,
                        pvalue,
                        is_correct_order,
                        r,
                    ]
                )

    return out


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)
    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")
    except FileNotFoundError:
        data = get_data()
        df = pd.DataFrame(
            data, columns=["Attack", "Model", "n", "pvalue", "is_correct_order", "r"]
        )
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10 * 1, 3 * 1),
    )

    plot_pvalues_cmp(
        None,
        df.loc[df.Attack == OURS],
        ax,
        hue="Model",
        xlabel="Number of samples in $\mathbf{Q}_{sus}$",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncols=1,
        title="Model",
        bbox_to_anchor=(1.15, 0.5),
    )
    plt.savefig(
        f"{PATH_TO_PLOTS}/pvalue_per_sample_{OURS}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

if __name__ == "__main__":
    main()

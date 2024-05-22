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
import matplotlib.colors

from itertools import product

from src import get_p_value
from experiments.utils import (
    set_plt,
    ATTACKS_NAME_MAPPING,
    MODELS_NAME_MAPPING,
    MODELS_ORDER,
    RESAMPLING_CNT,
    MODELS,
    RUN_ID,
    OURS,
)

set_plt()

NOISE_RATIOS = [0.0, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
NSAMPLES_PLOT = [30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
MODELS_TO_PLOT = ["LDM256", "DiT512", "U-ViT256-T2I"]
PATH_TO_SCORES = "out/scores"
PATH_TO_PLOTS = "experiments/out/noise_influence"

nsamples = np.array(NSAMPLES_PLOT)


def plot_noise_pvalue(model: str, df: pd.DataFrame, ax: plt.Axes, idx: int) -> None:
    sns.lineplot(
        data=df,
        x="noise_ratio",
        y="pvalue",
        hue="n",
        ax=ax,
        legend="full",
        palette="cool",
        hue_norm=matplotlib.colors.LogNorm(),
    )
    ax.set(
        title=model,
        yscale="log",
        ylim=[1e-5, 0.55],
    )
    ax.plot(
        df.noise_ratio,
        0.05 * np.ones_like(df.noise_ratio),
        "--",
        color="black",
        label="p-value: 0.05",
    )
    ax.plot(
        df.noise_ratio,
        0.01 * np.ones_like(df.noise_ratio),
        "--",
        color="green",
        label="p-value: 0.01",
    )
    if idx:
        ax.set(
            xlabel="Contamination ratio",
            ylabel="",
        )
    else:
        ax.set(
            xlabel="Contamination ratio",
            ylabel="p-value",
        )
    ax.legend().remove()


def get_data() -> list:
    out = []
    for attack, model, noise_ratio in tqdm(
        product(["cdi"], MODELS, NOISE_RATIOS)
    ):
        data = np.load(
            f"{PATH_TO_SCORES}/{model}_{attack}_{RUN_ID}.npz", allow_pickle=True
        )
        members_lower = data["metadata"][()]["members_lower"]
        n_eval_samples = data["metadata"][()]["n_samples_eval"]
        members = data["members"][:n_eval_samples]
        nonmembers = data["nonmembers"][:n_eval_samples]

        nsamples_run = nsamples[nsamples <= n_eval_samples]
        if noise_ratio:
            nsamples_run = nsamples_run[
                ((1 + noise_ratio) * nsamples_run).astype(int)
                == (1 + noise_ratio) * nsamples_run
            ]  # to ensure that we have precise ratio of "noise", e.g., 0.01 and 2 samples will not work out

        indices_total = np.arange(n_eval_samples)
        for n in nsamples_run:
            noise_samples = int(n * noise_ratio)
            for _ in range(RESAMPLING_CNT):
                indices = np.random.choice(
                    indices_total, n + noise_samples, replace=False
                )
                if noise_samples:
                    sample_members = np.concatenate(
                        [
                            members[indices[: n - noise_samples]],
                            nonmembers[indices[n : n + noise_samples]],
                        ]
                    )
                    sample_nonmembers = nonmembers[indices[:n]]
                else:
                    sample_members = members[indices]
                    sample_nonmembers = nonmembers[indices]

                assert len(sample_members) == n
                assert len(sample_nonmembers) == n
                pvalue, is_correct_order = get_p_value(
                    sample_members, sample_nonmembers, members_lower=members_lower
                )
                out.append(
                    [
                        ATTACKS_NAME_MAPPING[attack],
                        MODELS_NAME_MAPPING[model],
                        n,
                        noise_ratio,
                        pvalue,
                        is_correct_order,
                    ]
                )

    return out


def get_noise_pvalue_cmp(df: pd.DataFrame):
    fig, axs = plt.subplots(
        1, len(MODELS_TO_PLOT), figsize=(10 * len(MODELS_TO_PLOT), 5 * 1)
    )
    axs = axs.flatten()

    tmp_df = (
        df.loc[(df.Attack == OURS) & (df.n.isin(NSAMPLES_PLOT))]
        .groupby(["Model", "noise_ratio", "n"])
        .pvalue.mean()
        .reset_index()
    )
    for idx, (model, ax) in tqdm(enumerate(zip(MODELS_TO_PLOT, axs))):
        plot_noise_pvalue(model, tmp_df[tmp_df.Model == model], ax, idx)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncols=1,
        title="Number of samples in $\mathbf{Q}_{sus}$",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=18,
        title_fontsize=18,
    )

    plt.savefig(
        f"{PATH_TO_PLOTS}/noise_vs_pvalues.pdf", format="pdf", bbox_inches="tight"
    )

def get_fp_table(df: pd.DataFrame):
    df = (
        df.loc[(df.Attack == OURS) & (df.n == 10_000) & (df.noise_ratio.isin([0, 1]))]
        .groupby(["noise_ratio", "Model"])
        .pvalue.mean()
        .reset_index()
    )
    from math import log10, floor

    def find_exp(number) -> int:
        base10 = log10(abs(number))
        return floor(base10)

    df = df.pivot(index="noise_ratio", columns="Model", values="pvalue").applymap(
        lambda x: f"$10^{{{find_exp(x)}}}$" if (x < 0.1 and x != 0) else f"{x:.2f}"
    )[MODELS_ORDER]
    df.index.name = ""
    df.columns.name = ""
    df.to_latex(f"{PATH_TO_PLOTS}/fp_table.tex", escape=False)


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)
    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")
    except FileNotFoundError:
        data = get_data()
        df = pd.DataFrame(
            data,
            columns=[
                "Attack",
                "Model",
                "n",
                "noise_ratio",
                "pvalue",
                "is_correct_order",
            ],
        )
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)

    get_fp_table(df)
    get_noise_pvalue_cmp(df)


if __name__ == "__main__":
    main()

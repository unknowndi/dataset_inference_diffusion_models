import os
import sys

sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from itertools import product

from src import get_p_value
from src.attacks import get_datasets_clf
from experiments.utils import (
    set_plt,
    MODELS_NAME_MAPPING,
    RESAMPLING_CNT,
    RUN_ID,
    MODELS,
)

from typing import Tuple
from torch import Tensor as T
import multiprocessing as mp


set_plt()

PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "experiments/out/trainset_size_ablation"

nsamples = np.array(
    [n for n in range(20, 110, 10)]
    + [n for n in range(200, 1100, 100)]
    + [n for n in range(2000, 21000, 1000)]
)
n_eval_samples = 20_000
members_lower = False

trainset_sizes = np.array([200, 600, 1000, 2000])
trainset_sizes_single_proc = np.array(
    [6000]
)  # for some reason all cpus are used when running on 6k+ samples on all models

RETRAINING_CNT = 10

SIZES_TO_SHOW = [100, 300, 500, 1000, 3000, 5000]
COLORS = {size: color for size, color in zip(SIZES_TO_SHOW, sns.color_palette("tab10"))}
X_MEM_SIZE = "$\mathbf{X}_{mem}$ Size"


def get_datasets(
    members: T,
    nonmembers: T,
    train_size: int,
    eval_size: int,
):
    train_dataset, _, test_dataset = get_datasets_clf(
        members,
        nonmembers,
        5000,
        0,
        eval_size,
    )

    # first half is members, second half is nonmembers
    # we shuffle them to avoid any bias
    # we sample the same amount of members and nonmembers

    indices = np.random.permutation(len(train_dataset.data) // 2)
    indices = np.concatenate(
        [indices[: train_size // 2], 5000 + indices[: train_size // 2]]
    )

    train_dataset.data = train_dataset.data[indices]
    train_dataset.label = train_dataset.label[indices]

    ss = StandardScaler()

    train_dataset.data = ss.fit_transform(train_dataset.data)
    test_dataset.data = ss.transform(test_dataset.data)

    return train_dataset, test_dataset


def get_data_attack(
    members: np.ndarray, nonmembers: np.ndarray, train_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    train_dataset, test_dataset = get_datasets(
        members,
        nonmembers,
        train_size,
        n_eval_samples,
    )

    clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=None)
    clf.fit(train_dataset.data, train_dataset.label)
    preds = clf.predict_proba(test_dataset.data)[:, 1]
    members = preds[:n_eval_samples]
    nonmembers = preds[n_eval_samples:]

    return members, nonmembers


def _prun(model: str, sizes: list, pref: str):
    # aka process run
    data = np.load(
        f"{PATH_TO_FEATURES}/{model}_cdi_{RUN_ID}.npz", allow_pickle=True
    )

    members = torch.from_numpy(data["members"][:, 0, :])
    nonmembers = torch.from_numpy(data["nonmembers"][:, 0, :])
    out = []
    for train_size in tqdm(sizes):
        for _ in range(RETRAINING_CNT):
            members_scores, nonmembers_scores = get_data_attack(
                members, nonmembers, train_size
            )

            nsamples_run = nsamples[nsamples <= n_eval_samples]
            indices_total = np.arange(n_eval_samples)
            for n in nsamples_run:
                for _ in range(RESAMPLING_CNT):
                    indices = np.random.choice(indices_total, n, replace=False)
                    pvalue, is_correct_order = get_p_value(
                        members_scores[indices],
                        nonmembers_scores[indices],
                        members_lower=members_lower,
                    )
                    out.append(
                        [
                            train_size // 2,
                            MODELS_NAME_MAPPING[model],
                            n,
                            pvalue,
                            is_correct_order,
                        ]
                    )
    df = pd.DataFrame(
        out,
        columns=[
            X_MEM_SIZE,
            "Model",
            "n",
            "pvalue",
            "is_correct_order",
        ],
    )
    df.to_csv(f"{PATH_TO_PLOTS}/tmp/{pref}_{model}_pvalue_per_sample.csv", index=False)


def get_data() -> pd.DataFrame:
    out = []
    processes = []
    for model in MODELS:
        p = mp.Process(target=_prun, args=(model, trainset_sizes, "small"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for model in MODELS:
        _prun(model, trainset_sizes_single_proc, "big")

    for model, pref in product(MODELS, ["small", "big"]):
        df = pd.read_csv(f"{PATH_TO_PLOTS}/tmp/{pref}_{model}_pvalue_per_sample.csv")
        out.append(df)

    # we re-use data from features ablation
    df = pd.read_csv("experiments/out/features_ablation/pvalue_per_sample.csv")
    df = df.loc[df.Attack == f"All Features"]
    df = df.rename(columns={"Attack": X_MEM_SIZE})
    df[X_MEM_SIZE] = df[X_MEM_SIZE].map({f"All Features": 5000})
    df["is_correct_order"] = True
    out.append(df)

    out = pd.concat(out)
    return out


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    os.makedirs(f"{PATH_TO_PLOTS}/tmp", exist_ok=True)
    np.random.seed(42)

    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")
    except FileNotFoundError:
        df = get_data()
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)
    print("fin")


if __name__ == "__main__":
    main()

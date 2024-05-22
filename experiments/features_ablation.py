import os
import sys

sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import torch
from torch import Tensor as T
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    OURS,
)

from typing import Tuple


set_plt()

ATTACKS_FTS_INDICES = {
    "DL": [0],
    "SecMI": [1],
    "PIA": [2],
    "PIAN": [3],
    "MIA": np.arange(4),
    "GM": np.arange(4, 14),
    "ML": np.arange(14, 24),
    "NO": np.arange(24, 26),
    "MIA+GM": np.arange(14),
    "MIA+ML": np.concatenate([np.arange(4), np.arange(14, 24)]),
    "MIA+NO": np.concatenate([np.arange(4), np.arange(24, 26)]),
    "GM+ML": np.arange(4, 24),
    "GM+NO": np.concatenate([np.arange(4, 14), np.arange(24, 26)]),
    "ML+NO": np.arange(14, 26),
    "MIA+GM+ML": np.arange(24),
    "MIA+GM+NO": np.concatenate([np.arange(14), np.arange(24, 26)]),
    "MIA+ML+NO": np.concatenate([np.arange(4), np.arange(14, 26)]),
    "GM+ML+NO": np.arange(4, 26),
    "All Features": np.arange(26),
}

FTS_INDICES_NAME_MAPPING = {
    "DL": "denoising_loss",
    "SecMI": "secmi_stat",
    "PIA": "pia",
    "PIAN": "pian",
}

FTS_PRETTY_NAMES = {
    "DL": "Denoising Loss",
    "SecMI": "SecMI$_{stat}$",
    "PIA": "PIA",
    "PIAN": "PIAN",
    "GM": "Gradient Masking (GM)",
    "ML": "Multiple Loss (ML)",
    "NO": "Noise Optim (NO)",
    "All Features": f"All Features: {OURS}",
}

PATH_TO_SCORES = "out/scores"
PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "experiments/out/features_ablation"

nsamples = np.array(
    [n for n in range(20, 110, 10)]
    + [n for n in range(200, 1100, 100)]
    + [n for n in range(2000, 21000, 1000)]
)


def extract_metadata(metadata: dict) -> Tuple[bool, int]:
    members_lower = metadata["members_lower"]
    n_eval_samples = metadata["n_samples_eval"]
    return members_lower, n_eval_samples


def get_datasets(
    members: T,
    nonmembers: T,
    eval_size: int,
):
    train_dataset, _, test_dataset = get_datasets_clf(
        members,
        nonmembers,
        5000,
        0,
        eval_size,
    )

    ss = StandardScaler()

    train_dataset.data = ss.fit_transform(train_dataset.data)
    test_dataset.data = ss.transform(test_dataset.data)

    return train_dataset, test_dataset


def get_data_attack(
    members: np.ndarray, nonmembers: np.ndarray, n_eval_samples: int
) -> Tuple[np.ndarray, np.ndarray, bool, int]:

    train_dataset, test_dataset = get_datasets(
        members,
        nonmembers,
        n_eval_samples,
    )

    clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    clf.fit(train_dataset.data, train_dataset.label)
    preds = clf.predict_proba(test_dataset.data)[:, 1]
    members = preds[:n_eval_samples]
    nonmembers = preds[n_eval_samples:]

    return members, nonmembers


def get_data() -> list:
    out = []
    for model, attack in tqdm(product(MODELS, ATTACKS_FTS_INDICES.keys())):
        features_indices = ATTACKS_FTS_INDICES[attack]
        if len(features_indices) == 1:
            attack_name_raw = FTS_INDICES_NAME_MAPPING[attack]
            path = PATH_TO_SCORES
        else:
            attack_name_raw = "cdi"
            path = PATH_TO_FEATURES
        data = np.load(
            f"{path}/{model}_{attack_name_raw}_{RUN_ID}.npz", allow_pickle=True
        )
        members_lower, n_eval_samples = extract_metadata(data["metadata"][()])

        if len(features_indices) == 1:
            members = torch.from_numpy(data["members"])
            nonmembers = torch.from_numpy(data["nonmembers"])
        else:
            members = torch.from_numpy(data["members"][:, 0, features_indices])
            nonmembers = torch.from_numpy(data["nonmembers"][:, 0, features_indices])

            members, nonmembers = get_data_attack(members, nonmembers, n_eval_samples)

        nsamples_run = nsamples[nsamples <= n_eval_samples]
        indices_total = np.arange(n_eval_samples)
        for n in nsamples_run:
            for _ in range(RESAMPLING_CNT):
                indices = np.random.choice(indices_total, n, replace=False)
                pvalue, _ = get_p_value(
                    members[indices], nonmembers[indices], members_lower=members_lower
                )
                out.append([attack, MODELS_NAME_MAPPING[model], n, pvalue])
    return out

def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    os.makedirs(f"{PATH_TO_PLOTS}/tmp", exist_ok=True)
    np.random.seed(42)

    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")
    except FileNotFoundError:
        data = get_data()
        df = pd.DataFrame(data, columns=["Attack", "Model", "n", "pvalue"])
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)


if __name__ == "__main__":
    main()

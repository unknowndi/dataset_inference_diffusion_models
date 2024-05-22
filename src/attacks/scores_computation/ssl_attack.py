# https://arxiv.org/abs/2301.13188

from src.attacks import ScoreComputer
from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from scipy import stats


class SSLAttackComputer(ScoreComputer):
    def get_splits(
        self,
        data: T,
        train_samples: int,
        valid_samples: int,
    ) -> Tuple[T, T]:
        fit_size = train_samples + valid_samples
        test_size = data.size(0) - fit_size
        fit_data = data[test_size:]
        test_data = data[:test_size]
        return fit_data, test_data

    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        """
        Compute scores
        Inputs of shape (N_samples, 2, C, H, W)
        Outputs of shape (N_samples,)
        """

        n_samples = members.shape[0]
        members = F.normalize(members.reshape(n_samples, -1))
        nonmembers = F.normalize(nonmembers.reshape(n_samples, -1))

        fit_members, test_members = self.get_splits(
            members, self.config.train_samples, self.config.valid_samples
        )
        fit_nonmembers, test_nonmembers = self.get_splits(
            nonmembers, self.config.train_samples, self.config.valid_samples
        )

        gm = GaussianMixture(n_components=20, max_iter=1000, covariance_type="diag")
        gm.fit(fit_members)

        members_scores, nonmembers_scores = [], []
        for members, nonmembers in zip(
            (test_members, fit_members), (test_nonmembers, fit_nonmembers)
        ):
            members, nonmembers = gm.score_samples(members), gm.score_samples(
                nonmembers
            )
            members_scores.append(T(members))
            nonmembers_scores.append(T(nonmembers))

        return torch.concat(members_scores, dim=0), torch.concat(
            nonmembers_scores, dim=0
        )

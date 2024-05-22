# https://adam-dziedzic.com/static/assets/papers/dataset-inference-ssl.pdf
from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import pickle
from scipy import stats


class SSLAttackExtractor(FeatureExtractor):
    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        Simple L2 reconstruction loss of AE
        """
        images, classes = batch
        bs = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device)
        latents = self.model.encode(images)
        return latents

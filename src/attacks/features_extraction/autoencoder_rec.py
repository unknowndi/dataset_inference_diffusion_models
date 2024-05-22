from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple


class AutoEncReconLossThresholdExtractor(FeatureExtractor):
    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        Simple L2 reconstruction loss of AE
        """
        images, classes = batch
        bs = images.shape[0]
        images = images.to(self.device)

        losses = []
        for _ in range(self.attack_cfg.n_repetitions):
            latents = self.model.encode(images)
            images_recon = self.model.decode(latents)

            loss = torch.norm(
                images_recon.reshape(bs, -1) - images.reshape(bs, -1), dim=-1, p=2
            ).cpu()
            losses.append(loss)

        losses = torch.stack(losses, dim=1).unsqueeze(2)  # B, N_measurements, 1 feature
        return losses

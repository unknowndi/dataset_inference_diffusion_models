# https://arxiv.org/pdf/2302.01316.pdf

from src.attacks import ScoreComputer
from src.attacks.utils import get_datasets_clf
from torch import Tensor as T
import torch
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet

from typing import Tuple
from tqdm import tqdm
import copy


class SecMIStat(ScoreComputer):
    def compute_score(self, data: T) -> T:
        """
        Compute the score
        Output of shape (N_samples,)
        """
        x_det, x_step = data.permute(1, 0, 2, 3, 4)
        n = x_det.shape[0]
        return torch.norm(x_det.reshape(n, -1) - x_step.reshape(n, -1), dim=-1, p=2)

    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        """
        Compute scores
        Inputs of shape (N_samples, 2, C, H, W)
        Outputs of shape (N_samples,)
        """
        return self.compute_score(members), self.compute_score(nonmembers)


class SecMINN(ScoreComputer):
    def get_loaders(
        self, members: T, nonmembers: T
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, valid_dataset, test_dataset = get_datasets_clf(
            (members[:, 0] - members[:, 1]).abs(),
            (nonmembers[:, 0] - nonmembers[:, 1]).abs(),
            self.config.train_samples,
            self.config.valid_samples,
            self.config.n_samples_eval,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.attack_cfg.batch_size,
            shuffle=False,
            num_workers=self.attack_cfg.num_workers,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.attack_cfg.batch_size,
            shuffle=False,
            num_workers=self.attack_cfg.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.attack_cfg.batch_size,
            shuffle=False,
            num_workers=self.attack_cfg.num_workers,
        )

        return train_loader, valid_loader, test_loader

    def get_clf(self, train_loader: DataLoader, valid_loader: DataLoader) -> ResNet:
        model = resnet18(weights=None, num_classes=1)
        dim = next(iter(train_loader))[0].shape[1]
        if dim != 3:
            model.conv1 = torch.nn.Conv2d(
                dim, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        model.to(self.device)
        optim = torch.optim.SGD(
            model.parameters(), lr=self.attack_cfg.lr, momentum=0.9, weight_decay=5e-4
        )
        valid_acc_best_ckpt = None
        valid_acc_best = 0.0

        for _ in tqdm(range(self.attack_cfg.n_epoch), desc="SecMI_nn training"):
            self.train(model, optim, train_loader)
            valid_acc = self.eval(model, valid_loader)
            if valid_acc > valid_acc_best:
                valid_acc_best_ckpt = copy.deepcopy(model.state_dict())
                valid_acc_best = valid_acc

        model.load_state_dict(valid_acc_best_ckpt)
        model.eval()
        return model

    def train(self, model: ResNet, optimizer, loader: DataLoader) -> None:
        model.train()

        for data, label in loader:
            data = data.to(self.device)
            label = label.to(self.device).reshape(-1, 1)

            logit = model(data)

            loss = ((logit - label) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def eval(self, model: ResNet, loader: DataLoader) -> float:
        model.eval()

        total = 0
        correct = 0

        for data, label in loader:
            data, label = data.to(self.device), label.to(self.device).reshape(-1, 1)
            logit = model(data)
            total += data.size(0)

            logit[logit >= 0.5] = 1
            logit[logit < 0.5] = 0

            correct += (logit == label).sum().item()

        return correct / total

    @torch.no_grad()
    def compute_scores(self, model: ResNet, loader: DataLoader) -> T:
        """
        Compute scores
        Outputs of shape (N_samples,)
        """
        member_scores = []
        nonmember_scores = []
        for data, label in loader:
            logits = model(data.to(self.device))
            member_scores.append(logits[label == 1])
            nonmember_scores.append(logits[label == 0])

        member_scores = torch.concat(member_scores).reshape(-1)
        nonmember_scores = torch.concat(nonmember_scores).reshape(-1)
        return member_scores, nonmember_scores

    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        """
        Compute scores
        Inputs of shape (N_samples, 2, C, H, W)
        Outputs of shape (N_samples,)
        """
        train_loader, valid_loader, test_loader = self.get_loaders(members, nonmembers)
        clf = self.get_clf(train_loader, valid_loader)
        members_scores, nonmembers_scores = [], []
        for loader in [test_loader, valid_loader, train_loader]:
            members, nonmembers = self.compute_scores(clf, loader)
            members_scores.append(members)
            nonmembers_scores.append(nonmembers)

        return torch.concat(members_scores, dim=0), torch.concat(
            nonmembers_scores, dim=0
        )

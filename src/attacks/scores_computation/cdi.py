from src.attacks import ScoreComputer
from torch import Tensor as T
import torch


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.attacks.utils import MIDataset, get_datasets_clf

from typing import Tuple

classifiers = {
    "lr": LogisticRegression,
}


class CDIComputer(ScoreComputer):
    def fit_clf(self, train_dataset: MIDataset, seed: int):
        clf = classifiers[self.attack_cfg.clf](
            random_state=seed, **self.attack_cfg.kwargs
        )
        clf.fit(train_dataset.data, train_dataset.label)
        return clf

    def compute_score(self, data: T, clf) -> T:
        return torch.from_numpy(clf.predict_proba(data)[:, 1])

    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        train_dataset, valid_dataset, test_dataset = get_datasets_clf(
            members.reshape(members.shape[0], -1),
            nonmembers.reshape(members.shape[0], -1),
            self.config.train_samples,
            self.config.valid_samples,
            self.config.n_samples_eval,
        )
        ss = StandardScaler()

        # For the final evaluation we use all left-out samples as training samples

        train_dataset.data = torch.concat([train_dataset.data, valid_dataset.data])
        train_dataset.label = torch.concat([train_dataset.label, valid_dataset.label])

        train_dataset.data = torch.from_numpy(ss.fit_transform(train_dataset.data))
        test_dataset.data = torch.from_numpy(ss.transform(test_dataset.data))

        clf = self.fit_clf(train_dataset, self.config.seed)
        members_scores, nonmembers_scores = [], []

        for dataset in [test_dataset, train_dataset]:
            scores = self.compute_score(dataset.data, clf)

            members_scores.append(scores[: len(dataset) // 2])
            nonmembers_scores.append(scores[len(dataset) // 2 :])

        print(
            "train:",
            clf.score(train_dataset.data, train_dataset.label),
            "eval:",
            clf.score(test_dataset.data, test_dataset.label),
        )

        return torch.cat(members_scores), torch.cat(nonmembers_scores)

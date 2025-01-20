from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path: Path | str,
        targets_path: list[Path | str],
        factor_sizes,
        factor_discrete,
        ids: list[str] | None = None,
        normalize_sizes: list[int | None] | None = None,
    ):
        super().__init__()
        self._factor_sizes = factor_sizes
        self._factor_discrete = factor_discrete

        data = np.load(data_path)
        self.ids = sorted(data.files) if ids is None else ids
        self.lens = np.array([data[i].shape[0] for i in self.ids])
        self.feats = np.concatenate(
            [data[i] for i in self.ids],
            axis=0,
        ).astype(np.float32)
        del data

        if normalize_sizes is None:
            normalize_sizes = [None] * len(targets_path)
        fs = [
            (lambda x: x) if s is None else (lambda x: x / (s - 1)) for s in normalize_sizes
        ]

        ts = list(map(np.load, targets_path))
        self.targets = np.concatenate(
            [
                np.concatenate([f(t[i][:l, None]) for t, f in zip(ts, fs)], axis=1)
                for i, l in zip(self.ids, self.lens)
            ],
            axis=0,
        ).astype(np.float32)
        del ts

    def __len__(self) -> int:
        return self.feats.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.feats[idx]), torch.tensor(self.targets[idx])

    def get_feature_size(self) -> int:
        return self.feats.shape[-1]

    @property
    def normalized_targets(self):
        raise NotImplementedError


def make_datasets(cfg):
    gen = np.random.default_rng(cfg.seed)
    with np.load(cfg.representation_path) as reps:
        ids = reps.files
    ids = gen.permuted(ids).tolist()
    N = len(ids)

    split = cfg.split

    train_end_idx = int(split[0] * N)
    val_end_idx = train_end_idx + int(split[1] * N)

    splits = [train_end_idx, val_end_idx]

    train_dataset = CustomDataset(
        cfg.representation_path,
        cfg.factor_paths,
        cfg.factor_sizes,
        [True] * len(cfg.factor_sizes),
        ids=ids[: splits[0]],
        normalize_sizes=cfg.factor_sizes if cfg.normalize_targets else None,
    )

    val_dataset = CustomDataset(
        cfg.representation_path,
        cfg.factor_paths,
        cfg.factor_sizes,
        [True] * len(cfg.factor_sizes),
        ids=ids[splits[0] : splits[1]],
        normalize_sizes=cfg.factor_sizes if cfg.normalize_targets else None,
    )

    test_dataset = CustomDataset(
        cfg.representation_path,
        cfg.factor_paths,
        cfg.factor_sizes,
        [True] * len(cfg.factor_sizes),
        ids=ids[splits[1] :],
        normalize_sizes=cfg.factor_sizes if cfg.normalize_targets else None,
    )

    return train_dataset, val_dataset, test_dataset

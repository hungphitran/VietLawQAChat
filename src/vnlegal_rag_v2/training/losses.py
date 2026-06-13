from __future__ import annotations

import math
import random
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Sampler

from sentence_transformers.util import cos_sim


class MultiPositiveContrastiveLoss(nn.Module):
    """InfoNCE with multi-positive masking.

    Like MultipleNegativesRankingLoss but supports multiple positives per anchor.
    Uses a label tensor (group_id) to build a positive mask: all (query, doc) pairs
    with the same group_id are treated as positives.

    Data format: dataset with columns (query, positive, label=group_id)
    """

    def __init__(self, model, scale: float = 20.0):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, sentence_features, labels: torch.Tensor | None = None):
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        queries, docs = embeddings[0], embeddings[1]

        sim = cos_sim(queries, docs) * self.scale

        if labels is not None:
            positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        else:
            positive_mask = torch.eye(sim.size(0), device=sim.device)

        # Numerically stable: -log(sum(exp(pos)) / sum(exp(all)))
        max_sim = sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim - max_sim)

        pos_exp = (exp_sim * positive_mask).sum(dim=1)
        all_exp = exp_sim.sum(dim=1)

        loss = -torch.log(pos_exp / all_exp + 1e-8).mean()
        return loss


class GroupedBatchSampler(Sampler):
    """Batch sampler that groups rows by group_id into the same batch.

    Yields lists of indices (like PyTorch BatchSampler).
    Ensures all rows sharing the same group_id land in the same batch,
    so multi-positive loss can see all positives for a query together.
    """

    def __init__(
        self,
        group_ids: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 28,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            groups[gid].append(idx)

        self.groups = groups
        self.group_list = list(groups.keys())

    def __iter__(self):
        order = list(self.group_list)
        if self.shuffle:
            self.rng.shuffle(order)

        batches: list[list[int]] = []
        current_batch: list[int] = []

        for gid in order:
            indices = self.groups[gid]
            if len(current_batch) + len(indices) > self.batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = list(indices)
            else:
                current_batch.extend(indices)

        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        yield from batches

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        return math.ceil(total / self.batch_size)


class GroupedBatchSamplerFactory:
    """Pickle-able factory for GroupedBatchSampler.

    sentence-transformers trainer requires batch_sampler to be a callable
    that accepts (dataset, **kwargs) and returns a BatchSampler-like object.
    """

    def __init__(self, group_ids: list[int], batch_size: int, seed: int):
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.seed = seed

    def __call__(self, dataset, **kwargs):
        # Only use grouped sampler for the train dataset (size matches group_ids)
        # Fall back to default for eval dataset
        if len(dataset) == len(self.group_ids):
            return GroupedBatchSampler(
                group_ids=self.group_ids,
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed,
            )
        # Default: standard batch sampler for eval
        from torch.utils.data import BatchSampler, RandomSampler
        batch_size = kwargs.get("batch_size", self.batch_size)
        drop_last = kwargs.get("drop_last", False)
        return BatchSampler(
            RandomSampler(dataset),
            batch_size=batch_size,
            drop_last=drop_last,
        )

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_error(algorithm, x, y):
    if hasattr(algorithm, 'score'):
        if len(y.unique()) == 2:
            return 1 - algorithm.score(x, y)
        else:
            return np.mean((algorithm.predict(x) - y.squeeze().numpy()) ** 2)
    with torch.no_grad():
        if len(y.unique()) == 2:
            return algorithm.predict(x).gt(0).ne(y).float().mean().item()
        else:
            return (algorithm.predict(x) - y).pow(2).mean().item()


def compute_errors(model, envs):
    for split in envs.keys():
        if not bool(model.callbacks["errors"][split]):
            model.callbacks["errors"][split] = {
                key: [] for key in envs[split]["keys"]}

        for k, env in zip(envs[split]["keys"], envs[split]["envs"]):
            model.callbacks["errors"][split][k].append(
                compute_error(model, *env))

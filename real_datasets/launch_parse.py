import os

import numpy as np

from configs import LOG_FOLDER, get_parse_command

dataset = 'CUB'
gpu_idx = 0  # could be None if you want to use cpu

# Suppose we already trained the models for seeds 0, 1, 2, 3, 4,
# then we can parse these traind models by choosing log_seeds = np.arange(5)
train_log_seeds = np.arange(10)

# The training algorithms we want to parse
algos = ['ERM', 'reweight', 'groupDRO']

# load checkpoint with a model selection rule
# best: take the model at the epoch of largest worst-group validation accuracy
# best_avg_acc: take the model at the epoch of largest average-group validation accuracy
# last: take the trained model at the last epoch
model_selects = ['best', ]

log_dir = LOG_FOLDER

command = get_parse_command(dataset=dataset, algos=algos, model_selects=model_selects,
                            train_log_seeds=train_log_seeds, log_dir=log_dir, gpu_idx=gpu_idx,
                            parse_script='parse_features.py')
print('Command:', command)
os.system(command)

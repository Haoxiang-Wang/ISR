import os
import sys
import numpy as np
from config import *

num_data_seeds = 50  # we used 50 in the paper
num_model_seeds = 1
num_iterations = 10000  # we used 10000 in the paper
dim_inv = 5
dim_spu = 5
list_n_envs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
n_threads = 60 # -1 means to use all available CPU cores/threads; otherwise, use the specified number of threads
datasets = ["Example2", "Example2s", "Example3", "Example3s", "Example3_Modified", "Example3s_Modified"]
models = ["ISR_mean", "ISR_cov-flag", "ERM", "IRMv1", "IGA", "Oracle"]
for n_envs in list_n_envs:
    print('n_envs: {}'.format(n_envs))
    command = f"python sweep.py \
        --models {' '.join(models)}\
        --num_iterations {num_iterations} \
        --datasets {' '.join(datasets)} \
        --dim_inv {dim_inv} --dim_spu {dim_spu} \
        --n_envs {n_envs} \
        --num_data_seeds {num_data_seeds} --num_model_seeds {num_model_seeds} \
        --output_dir {RESULT_FOLDER}/nenvs/sweep_linear_nenvs={n_envs}_dinv={dim_inv}_dspu={dim_spu} \
        --n_threads {n_threads}"
    print('Command:', command)
    os.system(command)

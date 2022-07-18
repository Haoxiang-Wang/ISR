import os
from itertools import product
from tqdm import tqdm
from configs import get_train_command

gpu_idx = 0  # could be None if you want to use cpu

algos = ['ERM','reweight','groupDRO']
dataset = 'MultiNLI'  # could be 'CUB' (i.e., Waterbirds), 'CelebA' or 'MultiNLI'

# can add some suffix to the algo name to flag the version,
# e.g., with algo_suffix = "-my_version", the algo name becomes "ERM-my_version"
algo_suffix = ""

seeds = range(10)
for seed, algo in tqdm(list(product(seeds, algos)), desc='Experiments'):
    command = get_train_command(dataset=dataset, algo=algo, gpu_idx=gpu_idx, seed=seed,
                      save_best=True, save_last=True)
    print('Command:', command)
    os.system(command)

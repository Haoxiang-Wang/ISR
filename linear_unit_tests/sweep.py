import argparse
import os
import random
from glob import glob

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import datasets
import main
import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--models', nargs='+', default=[])
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--datasets', nargs='+', default=[])
    parser.add_argument('--dim_inv', type=int, default=5)
    parser.add_argument('--dim_spu', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_data_seeds', type=int, default=50)
    parser.add_argument('--num_model_seeds', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--callback', action='store_true')
    parser.add_argument('--n_threads', type=int, default=-1)
    parser.add_argument('--exp_name', type=str, default="default")
    args = vars(parser.parse_args())

    all_jobs = []
    if len(args["models"]) > 0:
        model_lists = args["models"]
    else:
        model_lists = models.MODELS.keys()
    if len(args["datasets"]):
        dataset_lists = args["datasets"]
    else:
        dataset_lists = datasets.DATASETS.keys()

    results_dirname = os.path.join(args["output_dir"], args["exp_name"] + "/")
    os.makedirs(results_dirname, exist_ok=True)
    for f in glob(f'{results_dirname}/*'):
        # remove all previous experiment results
        os.remove(f)

    for model in model_lists:
        for dataset in dataset_lists:
            for data_seed in range(args["num_data_seeds"]):
                for model_seed in range(args["num_model_seeds"]):
                    train_args = {
                        "model": model,
                        "num_iterations": args["num_iterations"],
                        "hparams": "random" if model_seed else "default",
                        "dataset": dataset,
                        "dim_inv": args["dim_inv"],
                        "dim_spu": args["dim_spu"],
                        "n_envs": args["n_envs"],
                        "num_samples": args["num_samples"],
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "output_dir": args["output_dir"],
                        "callback": args["callback"],
                        "exp_name": args["exp_name"],
                        "result_dir": results_dirname,
                    }

                    all_jobs.append(train_args)

    random.shuffle(all_jobs)

    print("Launching {} jobs...".format(len(all_jobs)))

    iterator = tqdm(all_jobs, desc="Jobs")
    if args["n_threads"] == 1:
        for job in iterator:
            main.run_experiment(job)
    else:
        Parallel(n_jobs=args["n_threads"])(delayed(main.run_experiment)(job) for job in iterator)

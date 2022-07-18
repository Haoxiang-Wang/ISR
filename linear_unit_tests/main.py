# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import hashlib
import pprint
import json
import os
import datasets
import models
import utils
from glob import glob

def run_experiment(args):
    # build directory name

    results_dirname = args['result_dir']

    # build file name
    md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    results_fname = os.path.join(results_dirname, md5_fname + ".jsonl")
    results_file = open(results_fname, "w")

    utils.set_seed(args["data_seed"])
    try:
        dataset = datasets.DATASETS[args["dataset"]](
            dim_inv=args["dim_inv"],
            dim_spu=args["dim_spu"],
            n_envs=args["n_envs"]
        )
    except:
        dataset_name, _ = args["dataset"].split('_')
        dataset = datasets.DATASETS[dataset_name](
            dim_inv=args["dim_inv"],
            dim_spu=args["dim_spu"],
            n_envs=args["n_envs"],
            rand_env_std=True
        )

    # Oracle trained on test mode (scrambled)
    train_split = "train" if args["model"] != "Oracle" else "test"

    # sample the envs
    if "ISR" in args["model"]:
        envs = {}
        for key_split, split in zip(("train", "validation", "test"),
                                    (train_split, train_split, "test")):
            envs[key_split] = []
            for env in dataset.envs:
                data = dataset.sample(
                    n=args["num_samples"],
                    env=env,
                    split=split)
                envs[key_split].append((data[0], data[1].flatten()))
    else:
        envs = {}
        for key_split, split in zip(("train", "validation", "test"),
                                    (train_split, train_split, "test")):
            envs[key_split] = {"keys": [], "envs": []}
            for env in dataset.envs:
                envs[key_split]["envs"].append(dataset.sample(
                    n=args["num_samples"],
                    env=env,
                    split=split)
                )
                envs[key_split]["keys"].append(env)

    # offsetting model seed to avoid overlap with data_seed
    utils.set_seed(args["model_seed"] + 1000)

    if "Example1" in args["dataset"]:
        regression = True
    else:
        regression = False
    # selecting model
    args["num_dim"] = args["dim_inv"] + args["dim_spu"]
    if "ISR" not in args["model"]:
        if args["model"] in ['ERM', 'Oracle']:
            model = models.MODELS[args["model"]](
                in_features=args["num_dim"],
                out_features=1,
                task=dataset.task,
                hparams=args["hparams"],
                regression=regression
            )
        else:
            model = models.MODELS[args["model"]](
                in_features=args["num_dim"],
                out_features=1,
                task=dataset.task,
                hparams=args["hparams"],
            )
        # update this field for printing purposes
        args["hparams"] = model.hparams
    else:
        model_n, model_m = args["model"].split('_')
        model = models.MODELS[model_n](
            dim_inv=max(1, args["dim_inv"]),
            fit_method=model_m,
            regression=regression,
            hparams=args["hparams"],
            num_iterations=args["num_iterations"]
        )

    # fit the dataset
    if "ISR" in args["model"]:
        model.fit(envs['train'])
        args["hparams"] = model.hparams
    else:
        model.fit(
            envs=envs,
            num_iterations=args["num_iterations"],
            callback=args["callback"])

    # compute the train, validation and test errors
    for split in ("train", "validation", "test"):
        key = "error_" + split
        if "ISR" in args["model"]:
            for k_env, env in enumerate(envs[split]):
                env = env
                args[key + "_E" +
                     str(k_env)] = utils.compute_error(model, *env)
        else:
            for k_env, env in zip(envs[split]["keys"], envs[split]["envs"]):
                env = env
                args[key + "_" +
                     k_env] = utils.compute_error(model, *env)

    # write results
    results_file.write(json.dumps(args))
    results_file.close()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic invariances')
    parser.add_argument('--model', type=str, default="ERM")
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--hparams', type=str, default="default")
    parser.add_argument('--dataset', type=str, default="Example1")
    parser.add_argument('--dim_inv', type=int, default=5)
    parser.add_argument('--dim_spu', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--callback', action='store_true')
    parser.add_argument('--exp_name', type=str, default="default")
    parser.add_argument('--result_dir', type=str, default=None)
    args = parser.parse_args()

    pprint.pprint(run_experiment(vars(args)))

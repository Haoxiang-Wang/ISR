# Experiments on Linear Unit Tests

This folder is built on the [released code](https://github.com/facebookresearch/InvarianceUnitTests)
of [Linear Unit-Tests](https://arxiv.org/abs/2102.10867), a work of Facebook Research.

## Preparation

Before runing the experiments, please put a folder path (for saving the experiment results) in `config.py` (i.e. define the `RESULT_FOLDER` variable).

## Run Experiments

Run the script

```shell
python launch_exp.py
```

which trains and evaluates ISR-Mean/Cov and baseline algorithms on 6 linear benchmarks as reported in our paper. Notice that "Example3_Modified" and "Example3s_Modified" are variants we implemented (reported as Example-3' and Example-3s' in our paper). 

## Plot Results

Run the following script to plot the results:

```shell
python plot_results.py
```
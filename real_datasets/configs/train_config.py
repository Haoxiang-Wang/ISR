


TRAIN_COMMANDS = dict(
    CelebA=
    {
        "ERM": "-s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.01 --lr 0.0001 "
               "--batch_size 128 --n_epochs 50",  # ERM
        "groupDRO": "-s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 3",
        # groupDRO
        "reweight": "-s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 "
                    "--batch_size 128 --n_epochs 50 --reweight_groups"  # reweight,
    }
    ,
    CUB={
        "ERM": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.1 --lr 0.0001 '
               '--batch_size 128 --n_epochs 300',  # ERM
        "reweight": '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 1 --lr 1e-05 '
                    '--batch_size 128 --n_epochs 300 --reweight_groups',  # reweight
        "groupDRO": ' -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 1 --lr 1e-05 '
                    '--batch_size 128 --n_epochs 300 --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 2',
        # groupDRO
    },
    MultiNLI={
        "ERM": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr '
               f'2e-05 --batch_size 32 --n_epochs 3',  # ERM
        "groupDRO": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups --robust --alpha 0.01 --gamma 0.1 '
                    '--generalization_adjustment 0',  # groupDRO
        "reweight": '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr '
                    f'2e-05 --batch_size 32 --n_epochs 3 --reweight_groups',  # reweight
    }
)


def get_train_command(dataset: str, algo: str, gpu_idx: int = None, train_script: str = 'run_expt.py',
                      algo_suffix: str = '',seed:int=None,save_best:bool=True,save_last:bool=True):
    prefix = f'CUDA_VISIBLE_DEVICES={gpu_idx}' if gpu_idx is not None else ''
    suffix = f' --algo_suffix {algo_suffix}' if algo_suffix else ''
    if save_best:
        suffix += ' --save_best'
    if save_last:
        suffix += ' --save_last'
    seed = 0 if seed is None else seed
    args_command = TRAIN_COMMANDS[dataset][algo]
    command = f"{prefix} python {train_script} {args_command} --seed {seed} {suffix}"
    return command
PARSE_COMMANDS = dict(
    MultiNLI=['-s', 'confounder', '-d', 'MultiNLI', '-t', 'gold_label_random',
              '-c', 'sentence2_has_negation', '--batch_size', '32', '--model', 'bert',
              '--n_epochs', '3', ],
    CelebA=['-d', 'CelebA', '-t', 'Blond_Hair', '-c', 'Male', '--model', 'resnet50',
            '--weight_decay', '0.01', '--lr', '0.0001',
            "--batch_size", '128', '--n_epochs', '50'],
    CUB=['-d', 'CUB', '-t', 'waterbird_complete95', '-c', 'forest2water2',
                '--model', 'resnet50', '--weight_decay', '0.1', '--lr', '0.0001',
                '--batch_size', '128', '--n_epochs', '300']
)


def get_parse_command(dataset, algos, train_log_seeds, model_selects,
                      log_dir: str, gpu_idx=None, parse_script='parse_features.py',
                      ):
    prefix = f'CUDA_VISIBLE_DEVICES={gpu_idx} ' if gpu_idx is not None else ''
    main_args = ' '.join(PARSE_COMMANDS[dataset])
    parse_algos = ' '.join(algos)
    parse_seeds = ' '.join(map(str, train_log_seeds))
    parse_model_selects = ' '.join(model_selects)
    parse_args = f" --parse_dir {log_dir} --parse_algos {parse_algos} --parse_seeds {parse_seeds} --parse_model_selects {parse_model_selects}"
    command = f'{prefix} python {parse_script} {main_args} {parse_args}'
    return command

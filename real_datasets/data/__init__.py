import os
from .confounder_utils import prepare_confounder_data
from .label_shift_utils import prepare_label_shift_data
from configs import DATA_FOLDER
root_dir = DATA_FOLDER

dataset_attributes = {
    'CelebA': {
        'root_dir': 'celebA_v1.0'
    },
    'CUB': {
        'root_dir': 'waterbirds'
    },
    'CIFAR10': {
        'root_dir': 'CIFAR10/data'
    },
    'MultiNLI': {
        'root_dir': 'multinli'
    }
}

for dataset in dataset_attributes:
    dataset_attributes[dataset]['root_dir'] = os.path.join(root_dir, dataset_attributes[dataset]['root_dir'])

shift_types = ['confounder', 'label_shift_step']


def prepare_data(args, train, return_full_dataset=False, train_transform=None, eval_transform=None):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    if args.shift_type == 'confounder':
        return prepare_confounder_data(args, train, return_full_dataset, train_transform=train_transform,
                                       eval_transform=eval_transform)
    elif args.shift_type.startswith('label_shift'):
        assert not return_full_dataset
        return prepare_label_shift_data(args, train)


def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(
            f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(
            f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(
                f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')

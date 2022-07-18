import argparse
import os
import pickle
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm

from configs.model_config import model_attributes
from data import dataset_attributes, shift_types, prepare_data
from utils.train_utils import check_args
from utils.train_utils import set_seed

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('-d', '--dataset',
                    choices=dataset_attributes.keys(), required=True)
parser.add_argument('-s', '--shift_type',
                    choices=shift_types, default='confounder')
# Confounders
parser.add_argument('-t', '--target_name')
parser.add_argument('-c', '--confounder_names', nargs='+')
# Resume?
parser.add_argument('--resume', default=False, action='store_true')
# Label shifts
parser.add_argument('--minority_fraction', type=float)
parser.add_argument('--imbalance_ratio', type=float)
# Data
parser.add_argument('--fraction', type=float, default=1.0)
parser.add_argument('--root_dir', default=None)
parser.add_argument('--reweight_groups', action='store_true', default=False)
parser.add_argument('--augment_data', action='store_true', default=False)
parser.add_argument('--val_fraction', type=float, default=0.1)
# Objective
parser.add_argument('--robust', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--generalization_adjustment', default="0.0")
parser.add_argument('--automatic_adjustment',
                    default=False, action='store_true')
parser.add_argument('--robust_step_size', default=0.01, type=float)
parser.add_argument('--use_normalized_loss',
                    default=False, action='store_true')
parser.add_argument('--btl', default=False, action='store_true')
parser.add_argument('--hinge', default=False, action='store_true')

# Model
parser.add_argument(
    '--model',
    choices=model_attributes.keys(),
    default='resnet50')
parser.add_argument('--train_from_scratch', action='store_true', default=False)

# Optimization
parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--minimum_variational_weight', type=float, default=0)
# Misc
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--show_progress', default=False, action='store_true')
parser.add_argument('--log_dir', default='/data/common/inv-feature/logs/')
parser.add_argument('--log_every', default=1e8, type=int)
parser.add_argument('--save_step', type=int, default=1e8)
parser.add_argument('--save_best', action='store_true', default=False)
parser.add_argument('--save_last', action='store_true', default=False)

parser.add_argument('--parse_algos', nargs='+',
                    default=['ERM', 'groupDRO', 'reweight'])
parser.add_argument('--parse_model_selects', nargs='+',
                    default=['best', 'best_avg_acc', 'last'],
                    help='best is based on worst-group validation accuracy.')
parser.add_argument('--parse_seeds', nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
parser.add_argument(
    '--parse_dir', default='/data/common/inv-feature/logs/', type=str)

args = parser.parse_args()
check_args(args)
if args.model == 'bert':
    args.max_grad_norm = 1.0
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0

if args.robust:
    algo = 'groupDRO'
elif args.reweight_groups:
    algo = 'reweight'
else:
    algo = 'ERM'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)
# Data
# Test data for label_shift_step is not implemented yet
test_data = None
test_loader = None
if args.shift_type == 'confounder':
    train_data, val_data, test_data = prepare_data(args, train=True)
elif args.shift_type == 'label_shift_step':
    train_data, val_data = prepare_data(args, train=True)

loader_kwargs = {'batch_size': args.batch_size,
                 'num_workers': 4, 'pin_memory': True}
train_loader = train_data.get_loader(
    train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
val_loader = val_data.get_loader(
    train=False, reweight_groups=None, **loader_kwargs)
if test_data is not None:
    test_loader = test_data.get_loader(
        train=False, reweight_groups=None, **loader_kwargs)

data = {}
data['train_loader'] = train_loader
data['val_loader'] = val_loader
data['test_loader'] = test_loader
data['train_data'] = train_data
data['val_data'] = val_data
data['test_data'] = test_data
n_classes = train_data.n_classes

# Initialize model
pretrained = not args.train_from_scratch

if model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
    assert pretrained
    # Load precomputed features
    d = train_data.input_size()[0]
    model = nn.Linear(d, n_classes)
    model.has_aux_logits = False
elif args.model == 'resnet50':
    model = torchvision.models.resnet50(pretrained=pretrained)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
elif args.model == 'resnet34':
    model = torchvision.models.resnet34(pretrained=pretrained)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
elif args.model == 'wideresnet50':
    model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
elif args.model == 'bert':
    assert args.dataset == 'MultiNLI'

    from transformers import BertConfig, BertForSequenceClassification

    config_class = BertConfig
    model_class = BertForSequenceClassification

    config = config_class.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        finetuning_task='mnli')
    model = model_class.from_pretrained(
        'bert-base-uncased',
        from_tf=False,
        config=config)
else:
    raise ValueError('Model not recognized.')

model = model.to(device)

if not args.model.startswith('bert'):
    encoder = torch.nn.Sequential(
        *(list(model.children())[:-1] + [torch.nn.Flatten()]))
    output_layer = model.fc


def process_batch(model, x, y=None, g=None, bert=True):
    if bert:
        input_ids = x[:, :, 0]
        input_masks = x[:, :, 1]
        segment_ids = x[:, :, 2]
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids,
        )
        pooled_output = outputs[1]
        logits = model.classifier(pooled_output)
        result = {'feature': pooled_output.detach().cpu().numpy(),
                  'pred': np.argmax(logits.detach().cpu().numpy(), axis=1),
                  }
    else:
        features = encoder(x)
        logits = output_layer(features)
        result = {'feature': features.detach().cpu().numpy(),
                  'pred': np.argmax(logits.detach().cpu().numpy(), axis=1),
                  }
    if y is not None:
        result['label'] = y.detach().cpu().numpy()
    if g is not None:
        result['group'] = g.detach().cpu().numpy()
    return result


for algo, model_select, seed in tqdm(list(product(args.parse_algos, args.parse_model_selects, args.parse_seeds)),
                                     desc='Iter'):
    print('Current iter:', algo, model_select, seed)
    save_dir = f'{args.parse_dir}/{args.dataset}/{algo}/s{seed}/'
    if not os.path.exists(save_dir):
        continue
    model.load_state_dict(torch.load(save_dir + f'/{model_select}_model.pth',
                                     map_location='cpu').state_dict())

    model.eval()

    # save the last linear layer (classifier head)
    if 'bert' in type(model).__name__.lower():
        weight = model.classifier.weight.detach().cpu().numpy()
        bias = model.classifier.bias.detach().cpu().umpy()
    elif 'resnet' in type(model).__name__.lower():
        weight = model.fc.weight.detach().cpu().numpy()
        bias = model.fc.bias.detach().cpu().numpy()
    else:
        raise ValueError(f'Unknown model type: {type(model)}')
    pickle.dump({'weight': weight, 'bias': bias}, open(
        save_dir + f'/{model_select}_clf.p', 'wb'))

    # save parsed features
    for split, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
        results = []
        fname = model_select + '_' + f'{split}_data.p'
        if os.path.exists(save_dir + '/' + fname):
            continue
        with torch.set_grad_enabled(False):
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch = tuple(t.to(device) for t in batch)
                x = batch[0]
                y = batch[1]
                g = batch[2]
                if args.model.startswith("bert"):
                    result = process_batch(model, x, y, g, bert=True)
                else:
                    result = process_batch(model, x, y, g, bert=False)
                results.append(result)
        parsed_data = {}
        for key in results[0].keys():
            parsed_data[key] = np.concatenate(
                [result[key] for result in results])

        pickle.dump(parsed_data, open(save_dir + '/' + fname, 'wb'))

        del results
        del parsed_data

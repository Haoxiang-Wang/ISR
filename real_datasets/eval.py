import argparse
import os
import pickle
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from configs import DATA_FOLDER
from isr import ISRClassifier, check_clf
from utils.eval_utils import extract_data, save_df, measure_group_accs, load_data, group2env

warnings.filterwarnings('ignore')  # filter out Pandas append warnings


def eval_ISR(args, train_data=None, val_data=None, test_data=None, log_dir=None):
    if (train_data is None) or (val_data is None) or (test_data is None) or (log_dir is None):
        train_data, val_data, test_data, log_dir = load_data(args)
    train_gs = train_data['group']
    n_train = len(train_gs)
    groups, counts = np.unique(train_data['group'], return_counts=True, axis=0)
    n_groups = len(groups)
    n_classes = len(np.unique(train_data['label']))
    # we do this because the original group is defined by (class * attribute)
    n_spu_attr = n_groups // n_classes
    assert n_spu_attr >= 2
    assert n_groups % n_classes == 0

    zs, ys, gs, preds = extract_data(train_data)

    test_zs, test_ys, test_gs, test_preds = extract_data(
        test_data, )
    val_zs, val_ys, val_gs, val_preds = extract_data(
        val_data, )

    if args.algo == 'ERM' or args.no_reweight:
        # no_reweight: do not use reweightning in the ISR classifier even if the args.algo is 'reweight' or 'groupDRO'
        sample_weight = None
    else:
        sample_weight = np.ones(n_train)
        for group, count in zip(groups, counts):
            sample_weight[train_gs == group] = n_train / n_groups / count
        if args.verbose:
            print('Computed non-uniform sample weight')

    df = pd.DataFrame(
        columns=['dataset', 'algo', 'seed', 'ckpt', 'split', 'method', 'clf_type', 'C', 'pca_dim', 'd_spu', 'ISR_class',
                 'ISR_scale', 'env_label_ratio'] +
                [f'acc-{g}' for g in groups] + ['worst_group', 'avg_acc', 'worst_acc', ])
    base_row = {'dataset': args.dataset, 'algo': args.algo,
                'seed': args.seed, 'ckpt': args.model_select, }

    # Need to convert group labels to env labels (i.e., spurious-attribute labels)
    es, val_es, test_es = group2env(gs, n_spu_attr), group2env(val_gs, n_spu_attr), group2env(test_gs, n_spu_attr)

    # eval_groups = np.array([0] + list(range(n_groups)))
    method = f'ISR-{args.ISR_version.capitalize()}'
    if args.no_reweight and (not args.use_orig_clf) and args.algo != 'ERM':
        method += '_noRW'
    if args.use_orig_clf:
        ckpt = pickle.load(open(log_dir + f'/{args.model_select}_clf.p', 'rb'))
        orig_clf = check_clf(ckpt, n_classes=n_classes)
        # Record original val accuracy:
        for (split, eval_zs, eval_ys, eval_gs) in [('val', val_zs, val_ys, val_gs),
                                                   ('test', test_zs, test_ys, test_gs)]:
            eval_group_accs, eval_worst_acc, eval_worst_group = measure_group_accs(orig_clf, eval_zs, eval_ys, eval_gs,
                                                                                   include_avg_acc=True)
            row = {**base_row, 'split': split, 'method': 'orig', **eval_group_accs, 'clf_type': 'orig',
                   'worst_acc': eval_worst_acc, 'worst_group': eval_worst_group}
            df = df.append(row, ignore_index=True)
        args.n_components = -1
        given_clf = orig_clf
        clf_type = 'orig'
    else:
        given_clf = None
        clf_type = 'logistic'

    if args.env_label_ratio < 1:
        rng = np.random.default_rng()
        # take a subset of training data
        idxes = rng.choice(len(zs), size=int(
            len(zs) * args.env_label_ratio), replace=False)
        zs, ys, gs, es = zs[idxes], ys[idxes], gs[idxes], es[idxes]

    np.random.seed(args.seed)
    # Start ISR
    ISR_classes = np.arange(
        n_classes) if args.ISR_class is None else [args.ISR_class]

    clf_kwargs = dict(C=args.C, max_iter=args.max_iter, random_state=args.seed)
    if args.ISR_version == 'mean': args.d_spu = n_spu_attr - 1

    isr_clf = ISRClassifier(version=args.ISR_version, pca_dim=args.n_components, d_spu=args.d_spu,
                            clf_type='LogisticRegression', clf_kwargs=clf_kwargs, )

    isr_clf.fit_data(zs, ys, es, n_classes=n_classes, n_envs=n_spu_attr)

    for ISR_class, ISR_scale in tqdm(list(product(ISR_classes, args.ISR_scales)), desc='ISR iter', leave=False):

        isr_clf.set_params(chosen_class=ISR_class, spu_scale=ISR_scale)

        if args.ISR_version == 'mean':
            isr_clf.fit_isr_mean(chosen_class=ISR_class, )
        elif args.ISR_version == 'cov':
            isr_clf.fit_isr_cov(chosen_class=ISR_class, )
        else:
            raise ValueError('Unknown ISR version')

        isr_clf.fit_clf(zs, ys, given_clf=given_clf, sample_weight=sample_weight)
        for (split, eval_zs, eval_ys, eval_gs) in [('val', val_zs, val_ys, val_gs),
                                                   ('test', test_zs, test_ys, test_gs)]:
            group_accs, worst_acc, worst_group = measure_group_accs(
                isr_clf, eval_zs, eval_ys, eval_gs, include_avg_acc=True)
            row = {**base_row, 'split': split, 'method': method, 'clf_type': clf_type, 'ISR_class': ISR_class,
                   'ISR_scale': ISR_scale, 'd_spu': args.d_spu, **group_accs, 'worst_group': worst_group,
                   'worst_acc': worst_acc, 'env_label_ratio': args.env_label_ratio}
            if not args.use_orig_clf:
                row.update({'C': args.C, 'pca_dim': args.n_components, })
            df = df.append(row, ignore_index=True)

    if args.verbose:
        print('Evaluation result')
        print(df)
    if not args.no_save:
        Path(args.save_dir).mkdir(parents=True,
                                  exist_ok=True)  # make dir if not exists
        save_df(df, os.path.join(args.save_dir,
                                 f'{args.dataset}_results{args.file_suffix}.csv'), subset=None, verbose=args.verbose)
    return df


def parse_args(args: list = None, specs: dict = None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_dir', type=str,
                           default=DATA_FOLDER)
    argparser.add_argument('--algo', type=str, default='ERM',
                           choices=['ERM', 'groupDRO', 'reweight'])
    argparser.add_argument(
        '--dataset', type=str, default='CelebA', choices=['CelebA', 'MultiNLI', 'CUB'])
    argparser.add_argument('--model_select', type=str,
                           default='best', choices=['best', 'best_avg_acc', 'last'])

    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--n_components', type=int, default=100)
    argparser.add_argument('--C', type=float, default=1)
    argparser.add_argument('--ISR_version', type=str, default='mean', choices=['mean', 'cov'])
    argparser.add_argument('--ISR_class', type=int, default=None,
                           help='None means enumerating over all classes.')
    argparser.add_argument('--ISR_scales', type=float,
                           nargs='+', default=[0, 0.5])
    argparser.add_argument('--d_spu', type=int, default=-1)
    argparser.add_argument('--save_dir', type=str, default='logs/')
    argparser.add_argument('--no_save', default=False, action='store_true')
    argparser.add_argument('--verbose', default=False, action='store_true')

    argparser.add_argument('--use_orig_clf', default=False,
                           action='store_true', help='Original Classifier only')
    argparser.add_argument('--env_label_ratio', default=1,
                           type=float, help='ratio of env label')
    argparser.add_argument('--feature_file_prefix', default='',
                           type=str, help='Prefix of the feature files to load')
    argparser.add_argument('--max_iter', default=1000, type=int,
                           help='Max iterations for the logistic solver')
    argparser.add_argument('--file_suffix', default='', type=str, )
    argparser.add_argument('--no_reweight', default=False, action='store_true',
                           help='No reweighting for ISR classifier on reweight/groupDRO features')
    config = argparser.parse_args(args=args)
    config.__dict__.update(specs)
    return config


if __name__ == '__main__':
    args = parse_args()
    eval_ISR(args)

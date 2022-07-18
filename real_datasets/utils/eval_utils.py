import os
import pickle

import numpy as np
import pandas as pd


def extract_data(data, transform=None, ):
    zs, ys, preds, gs = data['feature'], data['label'], data.get(
        'pred', None), data['group']
    if transform is not None:
        zs = transform(zs)

    return zs, ys, gs, preds


def check_row_exist_in_df(row, df=None, df_path=None):
    # check if a row exists in a dataframe
    if df is None:
        if not os.path.exists(df_path): return False
        df = pd.read_csv(df_path)
    arrays = []
    special_cols = []
    for col, val in row.items():
        if isinstance(val, list) or isinstance(val, tuple) or isinstance(val, np.ndarray):
            special_cols.append(col)
            continue
        arrays.append(df[col].values == val)
    exist_rows = np.prod(arrays, axis=0)
    n_exist = np.sum(exist_rows)
    if n_exist == 0 or len(special_cols) == 0:
        return n_exist > 0
    else:
        for col in special_cols:
            elements = np.array(row[col])
            if np.isin(elements, df.loc[exist_rows][col].values).mean() < 1:
                return False
            else:
                continue
        return True


def save_df(df, save_path, subset=None, verbose=False, drop_duplicates=True):
    if os.path.exists(save_path):
        orig_df = pd.read_csv(save_path)
        df = pd.concat([orig_df, df])
        if drop_duplicates:
            df = df.drop_duplicates(subset=subset,
                                    keep='last',
                                    ignore_index=True)
    df.to_csv(save_path, index=False)
    if verbose:
        print("Saved to", save_path)


def measure_group_accs(clf, zs, ys, gs, include_avg_acc=True):
    accs = {}
    if include_avg_acc:
        accs['avg_acc'] = clf.score(zs, ys)
    worst_group = None
    worst_acc = np.inf
    for g in np.unique(gs):
        g_idx = gs == g
        acc = clf.score(zs[g_idx], ys[g_idx])
        accs[f'acc-{int(g)}'] = acc
        if acc < worst_acc:
            worst_group = g
            worst_acc = acc
    return accs, worst_acc, worst_group


def group2env(groups, n_envs):
    # if the group is defined by id_class*n_envs+id_env,
    # this function can convert it to id_env
    return groups % n_envs


def load_data(args):
    log_dir = os.path.join(
        args.root_dir, f'{args.dataset}/{args.algo}/s{args.seed}/')
    prefix = args.feature_file_prefix + args.model_select + '_'
    if not os.path.exists(log_dir + f'/{prefix}train_data.p'):
        raise ValueError(f"No parsed {prefix}train_data.p at {log_dir}")

    train_data = pickle.load(open(log_dir + f'/{prefix}train_data.p', 'rb'))
    val_data = pickle.load(open(log_dir + f'/{prefix}val_data.p', 'rb'))
    test_data = pickle.load(open(log_dir + f'/{prefix}test_data.p', 'rb'))
    return train_data, val_data, test_data, log_dir

def update_args(args,specs:dict):
    for k,v in specs.items():
        if hasattr(args,k):
            setattr(args,k,v)
        else:
            raise ValueError(f"No attribute {k} in args")
    return args

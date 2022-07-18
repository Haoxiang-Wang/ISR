import warnings

import matplotlib.pyplot as plt
import numpy as np

from collect_results import *
from config import *

warnings.filterwarnings("ignore")  # ignore warnings for df.append


def plot_nenvs(df_data, dirname, file_name='', save=True, block=False):
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)

    fig, axs = plt.subplots(1, 6, figsize=(18, 3.1))

    axs = axs.flat
    # Top: nenvs
    datasets = df_data["dataset"].unique()
    for id_d, dataset in zip(range(len(datasets)), datasets):
        df_d = df_data[df_data["dataset"] == dataset]
        models = df_d["model"].unique()
        legends = []
        for id_m, model in zip(range(len(models)), models):
            if id_m == 0:
                marker = 'o'
            elif id_m == 1:
                marker = 's'
            else:
                marker = ''
            df_d_m = df_d[df_d["model"] == model].sort_values(by="n_envs")
            legend, = axs[id_d].plot(df_d_m["n_envs"], df_d_m["mean"],
                                     color=f'C{id_m}',
                                     label=model, marker=marker,
                                     linewidth=2)
            top = (df_d_m["mean"] + df_d_m["std"] / 2).to_numpy()
            bottom = (df_d_m["mean"] - df_d_m["std"] / 2).to_numpy()
            xs = np.arange(2, 11).astype(float)
            axs[id_d].fill_between(xs, bottom, top, facecolor=f'C{id_m}', alpha=0.2)
            legends.append(legend)

        axs[id_d].set_xlabel('n_env', fontsize=13, labelpad=-1)
        axs[id_d].set_title(dataset, fontsize=13)
        axs[id_d].set_ylim(bottom=-0.05, top=0.55)
        axs[id_d].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        axs[id_d].set_xticks([2, 4, 6, 8, 10])
        if id_d != 0:
            axs[id_d].set_yticklabels([])
        axs[id_d].set_xlim(left=1.5, right=10.5)
    fig.tight_layout()
    axs[0].set_ylabel("Mean Error", fontsize=14)

    plt.legend(handles=legends,
               ncol=6,
               loc="lower center",
               bbox_to_anchor=(-2.3, -0.5), prop={'size': 13})

    if save:
        fig_dirname = "figs/" + dirname + '/'
        os.makedirs(fig_dirname, exist_ok=True)
        models = '_'.join(models)
        plt.savefig(fig_dirname + file_name + '.pdf',
                    format='pdf', bbox_inches='tight')
    if block:
        plt.show(block=False)
        input('Press to close')
        plt.close('all')


def build_df(dirname):
    print(dirname)
    df = pd.DataFrame(columns=['n_envs', 'dim_inv', 'dim_spu', 'dataset', 'model', 'mean', 'std'])
    for filename in glob.glob(os.path.join(dirname, "*.jsonl")):
        with open(filename) as f:
            dic = json.load(f)
            n_envs = dic["n_envs"]
            dim_inv = dic["dim_inv"]
            dim_spu = dic["dim_spu"]
            for dataset in dic["data"].keys():
                single_dic = {}
                for model in dic["data"][dataset].keys():
                    mean = dic["data"][dataset][model]["mean"]
                    std = dic["data"][dataset][model]["std"]
                    single_dic = dict(
                        n_envs=n_envs,
                        dim_inv=dim_inv,
                        dim_spu=dim_spu,
                        dataset=dataset,
                        model=model,
                        mean=mean,
                        std=std
                    )
                    # print(single_dic)
                    df = df.append(single_dic, ignore_index=True)
    return df


def process_results(dirname, exp_name, save_dirname):
    subdirs = [os.path.join(dirname, subdir, exp_name + '/') for subdir in os.listdir(dirname) if
               os.path.isdir(os.path.join(dirname, subdir))]
    for subdir in subdirs:
        print(subdir)
        table, table_avg, table_hparams, table_val, table_val_avg, df = build_table(subdir)

        # save table_val_avg
        save_dirname_avg = save_dirname + "avg/"
        os.makedirs(save_dirname_avg, exist_ok=True)
        results_filename = os.path.join(save_dirname_avg, 'avg_' + '_'.join(subdir.split('/')[-4:-1]) + ".jsonl")
        results_file = open(results_filename, "w")
        results_file.write(json.dumps(table_val_avg))
        results_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default=RESULT_FOLDER)
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    dirname_nenvs = "results_processed/nenvs/" + args.exp_name + "/"

    # construct averaged data
    if not args.load:
        process_results(dirname=args.dirname + "nenvs/", exp_name=args.exp_name, save_dirname=dirname_nenvs)

        # plot results for different number of envs
    df_nenvs = build_df(dirname_nenvs + "avg/")

    plot_nenvs(df_nenvs, dirname=args.dirname.split('/')[-1],
               file_name='results_nenvs_dimspu_' + args.exp_name,
               save=True, block=False)

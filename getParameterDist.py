import argparse
import math
import os
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt
from sympy import ceiling

from clustering_handler import perform_clustering, eval_clustering_supervised, eval_clustering_unsupervised
from configspace_handler import get_configspace
from data_handler import load_data
from subset_handler import load_subset


def get_score(metrics, score_type, worst):
    if score_type=="supervised":
        if worst:
            return 2
        else:
            return 2 - metrics["AMI"] - metrics["ARI"]
    elif score_type=="unsupervised":
        if worst:
            return 2
        else:
            return 2 - metrics["SilhouetteScore"] - metrics["DISCO5"]
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="complex9", type=str, help='Dataset')
    parser.add_argument('--size', default=0.5, type=float, help='Size of Dataset')
    parser.add_argument('--sampling', default="birch", type=str, help='Downsampling Strategy')
    parser.add_argument('--method', default="hdbscan", type=str, help='Clustering Method')
    parser.add_argument('--supervised', default=1, type=int, help='Use supervised scoring')
    parser.add_argument('--param_configs', default=50, type=int, help='Number of configs per parameter')
    parser.add_argument('--primary', default="min_samples", type=str, help='Primary parameter')
    parser.add_argument('--secondary', default="alpha", type=str, help='Secondary parameter')
    parser.add_argument('--tertiary', default="", type=str, help='Tertiary parameter')
    parser.add_argument('--primary_scale', default="sample_mult", type=str, help='Primary parameter scaling')
    parser.add_argument('--secondary_scale', default="", type=str, help='Secondary parameter scaling')
    parser.add_argument('--tertiary_scale', default="", type=str, help='Tertiary parameter scaling')
    args = parser.parse_args()
    args.supervised = args.supervised == 1
    full_data_points, full_labels = load_data(args.ds)
    configspace = get_configspace(args.method, args.ds, full_data_points)
    configspace_dict = dict(configspace)
    param_value_dict = {}
    for config_key in configspace_dict.keys():
        param = configspace_dict[config_key]
        default = param.default_value
        if "UniformFloatHyperparameter" in str(type(param)):
            lower = param.lower
            upper = param.upper
            if config_key == args.primary:
                if args.primary_scale == "sample_mult":
                    lower *= args.size
                    upper *= args.size
            if config_key == args.secondary:
                if args.secondary_scale == "sample_mult":
                    lower *= args.size
                    upper *= args.size
            if config_key == args.tertiary:
                if args.tertiary_scale == "sample_mult":
                    lower *= args.size
                    upper *= args.size

            if param.log:
                if param.lower == 0:
                    values = np.geomspace(0.00000000000001, upper, args.param_configs)
                else:
                    values = np.geomspace(lower, upper, args.param_configs)
            else:
                values = np.linspace(lower, upper, args.param_configs)
        elif "UniformIntegerHyperparameter" in str(type(param)):
            lower = param.lower
            upper = param.upper
            if config_key == args.primary:
                if args.primary_scale == "sample_mult":
                    lower *= args.size
                    upper *= args.size
            if config_key == args.secondary:
                if args.secondary_scale == "sample_mult":
                    lower *= args.size
                    upper *= args.size
            if config_key == args.tertiary:
                if args.tertiary_scale == "sample_mult":
                    lower *= args.size
                    upper *= args.size
            lower = math.ceil(lower)
            upper = math.ceil(upper)

            if param.log:
                if param.lower == 0:
                    values = np.unique(np.geomspace(0.00000000000001, upper, args.param_configs, dtype=int))
                else:
                    values = np.unique(np.geomspace(lower, upper, args.param_configs, dtype=int))
            else:
                values = np.unique(np.linspace(lower, upper, args.param_configs, dtype=int))
        elif "CategoricalHyperparameter" in str(type(param)):
            values = param.choices
        print(default)
        param_value_dict[config_key] = values

    if args.primary == "" and args.secondary == "":
        param_keys = list(configspace_dict.keys())
        args.primary = param_keys[0]
        if len(param_keys)>1:
            args.secondary = param_keys[1]
    print(args.primary, args.secondary)

    primary_values = param_value_dict[args.primary]
    if args.secondary != "":
        secondary_values = param_value_dict[args.secondary]
    else:
        secondary_values = [None]

    if args.tertiary != "":
        tertiary_values = param_value_dict[args.tertiary]
    else:
        tertiary_values = [None]

    score_type = ""
    if args.supervised:
        score_type = "supervised"
    else:
        score_type = "unsupervised"

    if not os.path.exists("param_plot"):
        os.makedirs("param_plot")
    if not os.path.exists("param_plot_dicts"):
        os.makedirs("param_plot_dicts")
    for seed in range(1):

        if args.size == 1.0:
            data_seed = 0
        else:
            data_seed = seed

        if args.size == 1.0:
            data_points, labels = load_data(args.ds)
        else:
            data_points, labels = load_subset(args.ds, args.size, args.sampling, data_seed)

        for tertiary_value in tertiary_values:
            all_scores = np.zeros((len(secondary_values), len(primary_values)))
            print(all_scores)
            i = 0
            for primary_value in primary_values:
                j = 0
                for secondary_value in secondary_values:
                    try:
                        cur_config = {args.primary: primary_value}
                        if args.secondary != "":
                            cur_config[args.secondary] = secondary_value
                        if args.tertiary != "":
                            cur_config[args.tertiary] = tertiary_value
                        start = default_timer()
                        clustering = perform_clustering(data_points, args.method, cur_config, seed)
                        time = float(default_timer() - start)
                        metrics = eval_clustering_supervised(clustering, labels)
                        metrics |= eval_clustering_unsupervised(clustering, data_points)
                        metrics |= {"clu_num": len(np.unique(clustering)),
                                    "clu_histogram": np.unique(clustering, return_counts=True)[1].tolist(), "Time": time}
                        print(cur_config, metrics)
                        score = get_score(metrics, score_type, False)
                        all_scores[j,i] = score
                    except:
                        print(cur_config, "failed")
                        score = get_score({}, score_type, True)
                        all_scores[j,i] = score

                    j +=1
                i+= 1
            worst_score = get_score({}, score_type, True)
            plt.figure(figsize=(10,10))
            plt.imshow(worst_score-all_scores, vmin = 0, vmax = worst_score)

            if len(primary_values) > 10:
                primary_indices = np.linspace(0,len(primary_values), endpoint=False, num=10, dtype=int)
                primary_labels = primary_values[primary_indices]
            else:
                primary_indices = range(len(primary_values))
                primary_labels = primary_values

            if "Float" in str(type(configspace_dict[args.primary])):
                ##https://stackoverflow.com/questions/8595973/truncate-to-three-decimals-in-python
                primary_labels = ['%.3f'%primary_label for primary_label in primary_labels]


            if len(secondary_values) > 10:
                secondary_indices = np.linspace(0,len(secondary_values), endpoint=False, num=10, dtype=int)
                secondary_labels = secondary_values[secondary_indices]
            else:
                secondary_indices = range(len(secondary_values))
                secondary_labels = secondary_values

            if "Float" in str(type(configspace_dict[args.secondary])):
                secondary_labels = ['%.3f'%secondary_label for secondary_label in secondary_labels]


            plt.xticks(primary_indices, primary_labels, rotation=45)
            plt.xlabel(args.primary)
            if args.secondary != "":
                plt.yticks(secondary_indices,  secondary_labels)
                plt.ylabel(args.secondary)
            if tertiary_value is not None:
                tertiary_label = tertiary_value
                if "Float" in str(type(configspace_dict[args.tertiary])):
                    tertiary_label = '%.3f' % tertiary_label
                title = str(args.tertiary) + ": " + str(tertiary_label) + " Seed: " + str(seed)
                plt.title(title)
            else:
                title = "Seed: " + str(seed)
                plt.title(title)
            np.save(f"param_plot_dicts/{args.ds}_{args.method}_{args.primary}_{args.primary_scale}_{args.secondary}_{args.secondary_scale}_{args.tertiary}_{args.tertiary_scale}_{tertiary_value}_{args.supervised}_{args.sampling}_{args.size}_{args.param_configs}_{data_seed}.npy", all_scores)

            figname = f"param_plot/{args.ds}_{args.method}_{args.primary}_{args.primary_scale}_{args.secondary}_{args.secondary_scale}_{args.tertiary}_{args.tertiary_scale}_{tertiary_value}_{args.supervised}_{args.sampling}_{args.size}_{args.param_configs}_{data_seed}.pdf"
            plt.savefig(figname)
            #plt.show()



import argparse
import copy
import math
import os
import traceback
from timeit import default_timer

from clustering_handler import eval_clustering_supervised, eval_clustering_unsupervised, perform_clustering
from configspace_handler import get_configspace
from data_handler import load_data
import numpy as np

from subset_handler import load_subset

# perform grid search on dataset and saves results in grid_evals
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="complex9", type=str, help='Dataset')
    parser.add_argument('--size', default=1.0, type=float, help='Size of Dataset')
    parser.add_argument('--sampling', default="random", type=str, help='Downsampling Strategy')
    parser.add_argument('--method', default="em", type=str, help='Clustering Method')
    parser.add_argument('--param_configs', default=100, type=int, help='Number of configs per parameter')
    parser.add_argument('--primary', default="n_components", type=str, help='Primary parameter')
    parser.add_argument('--secondary', default="init_params", type=str, help='Secondary parameter')
    parser.add_argument('--tertiary', default="covariance_type", type=str, help='Tertiary parameter')

    # relevant combinations:
    #   kmeans n_clusters init none
    #   dbscan eps min_samples none
    #   em n_components init_params covariance_type
    #   spectral n_clusters gamma none
    #   spectral n_clusters n_neighbors none
    #   spectral n_clusters gamma assign_labels_d
    #   spectral n_clusters n_neighbors assign_labels_d
    #   spectral n_clusters gamma assign_labels_c
    #   spectral n_clusters n_neighbors assign_labels_c
    #   agglomerative n_clusters linkage none
    #   agglomerative n_clusters n_neighbors linkage

    args = parser.parse_args()
    args.data_seed = None
    args.supervised = None
    print(args)
    full_data_points, full_labels = load_data(args.ds)
    configspace = get_configspace(args.method, args.ds, args)
    configspace_dict = dict(configspace)
    param_value_dict = {}
    for config_key in configspace_dict.keys():
        param = configspace_dict[config_key]
        default = param.default_value
        if config_key in [args.primary, args.secondary, args.tertiary]:
            if "UniformFloatHyperparameter" in str(type(param)):
                lower = param.lower
                upper = param.upper
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
            elif "Constant" in str(type(param)):
                values = default
        else:
            values = default
        param_value_dict[config_key] = values
        print(param)
    if args.method == "spectral" and (args.primary == "n_neighbors" or args.secondary == "n_neighbors"):
        param_value_dict["affinity"] = "nearest_neighbors"
    elif args.method == "spectral" and (args.primary == "gamma" or args.secondary == "gamma"):
        param_value_dict["affinity"] = "rbf"
    if args.method == "agglomerative" and (args.primary == "n_neighbors" or args.secondary == "n_neighbors"):
        param_value_dict["connectivity"] = "kneighbors_graph"

    if args.tertiary == "assign_labels_d":
        param_value_dict["assign_labels"] = "discretize"

    if args.tertiary == "assign_labels_c":
        param_value_dict["assign_labels"] = "cluster_qr"

    print(param_value_dict)

    primary_values = param_value_dict[args.primary]
    if args.secondary != "none":
        secondary_values = param_value_dict[args.secondary]
    else:
        secondary_values = [None]

    if args.tertiary != "none" and args.tertiary != "assign_labels_c" and args.tertiary != "assign_labels_d":
        tertiary_values = param_value_dict[args.tertiary]
    else:
        tertiary_values = [None]

    if not os.path.exists("grid_evals"):
        os.makedirs("grid_evals", exist_ok=True)

    if args.sampling in ["random", "lwc", "kcentroid"] and args.size < 1:
        data_seeds = range(3)
    else:
        data_seeds = [0]

    if args.method in ["kmeans", "em", "spectral"]:
        eval_seeds = range(5)
    else:
        eval_seeds = [0]

    eval_path = f'grid_evals/{args.ds}_{args.sampling}_{args.size}_{args.method}_{args.primary}_{args.secondary}_{args.tertiary}_{args.param_configs}.txt'
    eval_log_file = open(eval_path, 'w', buffering=1)

    h = 0
    for tertiary_value in tertiary_values:
        i = 0
        for primary_value in primary_values:
            j = 0
            for secondary_value in secondary_values:
                for data_seed in data_seeds:
                    if args.size == 1.0:
                        data_points, labels = load_data(args.ds)
                    elif args.size == 2.0:
                        data_points_regular, labels_regular = load_data(args.ds)
                        data_points = np.repeat(data_points_regular,2,axis=0)
                        labels = np.repeat(labels_regular,2,axis=0)
                        #print(data_points.shape, labels.shape)
                    else:
                        data_points, labels = load_subset(args.ds, args.size, args.sampling, data_seed)
                    cur_config = copy.deepcopy(param_value_dict)
                    if args.primary in cur_config.keys():
                        cur_config[args.primary] = primary_value
                    if args.secondary in cur_config.keys():
                        cur_config[args.secondary] = secondary_value
                    if args.tertiary in cur_config.keys():
                        cur_config[args.tertiary] = tertiary_value
                    for eval_seed in eval_seeds:
                        try:
                            start = default_timer()
                            clustering = perform_clustering(data_points, args.method, cur_config, eval_seed)
                            time = float(default_timer() - start)
                            metrics = eval_clustering_supervised(clustering, labels)
                            metrics |= eval_clustering_unsupervised(clustering, data_points)
                            metrics |= {"clu_num": len(np.unique(clustering)),
                                        "clu_histogram": np.unique(clustering, return_counts=True)[1].tolist(), "Time": time}
                            sup_score = metrics["AMI"] + metrics["ARI"]
                            unsup_score = metrics["SilhouetteScore"] + metrics["DISCO5"]
                            out_text = f"{primary_value};{secondary_value};{tertiary_value};{data_seed};{eval_seed};{cur_config};{metrics};{sup_score};{unsup_score};{i};{j};{h}\n"
                            eval_log_file.write(out_text)
                        except:
                            out_text = f"{primary_value};{secondary_value};{tertiary_value};{data_seed};{eval_seed};{cur_config};crashed;-2;-2;{i};{j};{h}\n"
                            eval_log_file.write(out_text)
                            print(traceback.format_exc())
                j+= 1
            i += 1
        h += 1
    eval_log_file.close()
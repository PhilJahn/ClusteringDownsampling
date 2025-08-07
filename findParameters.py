import argparse
import os
from timeit import default_timer

from ConfigSpace import Configuration
from smac import Scenario, HyperparameterOptimizationFacade

import util
import numpy as np

from clustering_handler import perform_clustering, eval_clustering_supervised, eval_clustering_unsupervised
from configspace_handler import get_configspace
from data_handler import load_data
from subset_handler import load_subset

global method
global supervised
global data_points
global labels
global performance_log_file


def clustering_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    start = default_timer()
    clustering = perform_clustering(data_points, method, config_dict, seed)
    time = float(default_timer() - start)
    metrics = eval_clustering_supervised(clustering, labels)
    metrics |= eval_clustering_unsupervised(clustering, data_points)
    metrics |= {"clu_num": len(np.unique(clustering)), "clu_histogram": np.unique(clustering, return_counts=True)[1].tolist(), "Time": time}
    if supervised:
        score = 2 - metrics["AMI"] - metrics["ARI"]
    else:
        score = 2 - metrics["SilhouetteScore"] - metrics["DISCO5"]
    performance_log_file.write(f"{config_dict}, {metrics}\n")
    return score



def run_parameter_estimation(args, seed, name):
    scenario = Scenario(get_configspace(method, args.ds, data_points), use_default_config=True,
                        n_trials=10000000, walltime_limit=args.budget, seed=seed, name=name)
    smac = HyperparameterOptimizationFacade(scenario, clustering_runner, overwrite=True)
    incumbent = smac.optimize()
    score = smac.runhistory.get_min_cost(incumbent)
    run_num = len(smac.runhistory.items())
    return incumbent, score, scenario, run_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="EEG Eye State", type=str, help='Dataset')
    parser.add_argument('--size', default=0.5, type=float, help='Size of Dataset')
    parser.add_argument('--sampling', default="kmeans", type=str, help='Downsampling Strategy')
    parser.add_argument('--method', default="dbscan", type=str, help='Clustering Method')
    parser.add_argument('--budget', default=600, type=int, help='SMAC AutoML Budget (in seconds)')
    parser.add_argument('--supervised', default=1, type=int, help='Use supervised scoring')
    args = parser.parse_args()
    args.supervised = args.supervised == 1

    method = args.method
    supervised = args.supervised
    sup_string = "supervised" if supervised else "unsupervised"

    if not os.path.exists("opt_logs"):
        os.makedirs("opt_logs")
    if not os.path.exists("param_logs"):
        os.makedirs("param_logs")

    if not os.path.exists(f"opt_logs/{args.ds}_{method}"):
        os.makedirs(f"opt_logs/{args.ds}_{method}")

    if not os.path.exists(f"param_logs/{args.ds}_{method}"):
        os.makedirs(f"param_logs/{args.ds}_{method}")

    if args.size == 1.0:
        parameter_log_file = open(
                    f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_full_{args.budget}.csv', 'w',
                    buffering=1)
    else:
        parameter_log_file = open(
            f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}.csv', 'w',
            buffering=1)

    for seed in range(5):
        if args.size == 1.0:
            data_seed = 0
        else:
            data_seed = seed
        if args.size == 1.0:
            data_points, labels = load_data(args.ds)
            performance_log_file = open(
                f'opt_logs/{args.ds}_{method}/log_{args.ds}_{method}_{sup_string}_full_{args.budget}_{seed}.txt', 'w',
                buffering=1)
        else:
            data_points, labels = load_subset(args.ds, args.size, args.sampling, data_seed)
            performance_log_file = open(
                f'opt_logs/{args.ds}_{method}/log_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}_{data_seed}_{seed}.txt', 'w',
                buffering=1)
        name = f"{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}_{data_seed}"
        incumbent, score, scenario, run_num = run_parameter_estimation(args, seed, name)
        parameter_log_file.write(f"{seed};{dict(incumbent)};{score};{run_num}\n")
        performance_log_file.close()



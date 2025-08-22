import argparse
import os
from timeit import default_timer

from ConfigSpace import Configuration
from smac import Scenario, HyperparameterOptimizationFacade, AlgorithmConfigurationFacade

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
    score_sum = 0
    metricss = []
    for seed in range(3): # take average of 3
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
        score_sum += score
        metricss.append(metrics)
    score = score_sum/5
    performance_log_file.write(f"{config_dict}, {score}\n")
    for metrics in metricss:
        performance_log_file.write(f"\t{config_dict}: {metrics}\n")
    return score



def run_parameter_estimation(args, seed, name):
    scenario = Scenario(get_configspace(method, args.ds, data_points), use_default_config=True,
                        n_trials=10000000, walltime_limit=args.budget, seed=seed, name=name, deterministic=True)
    smac = AlgorithmConfigurationFacade(scenario, clustering_runner, overwrite=True)
    incumbent = smac.optimize()
    score = smac.runhistory.get_min_cost(incumbent)
    run_num = len(smac.runhistory.items())
    return incumbent, score, scenario, run_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="complex9", type=str, help='Dataset')
    parser.add_argument('--size', default=1.0, type=float, help='Size of Dataset')
    parser.add_argument('--sampling', default="kmeans", type=str, help='Downsampling Strategy')
    parser.add_argument('--method', default="em", type=str, help='Clustering Method')
    parser.add_argument('--budget', default=60, type=int, help='SMAC AutoML Budget (in seconds)')
    parser.add_argument('--data_seed', default=0, type=int, help='Seed for dataset')
    parser.add_argument('--smac_seed', default=0, type=int, help='Seed for SMAC3')
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
                    f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_full_{args.budget}_{args.smac_seed}.csv', 'w',
                    buffering=1)
    else:
        parameter_log_file = open(
            f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}_{args.data_seed}_{args.smac_seed}.csv', 'w',
            buffering=1)

    if args.size == 1.0 and args.data_seed != 0:
        raise ValueError("data seeds only for subsampled data")
    if args.size == 1.0:
        data_points, labels = load_data(args.ds)
        performance_log_file = open(
            f'opt_logs/{args.ds}_{method}/log_{args.ds}_{method}_{sup_string}_full_{args.budget}_{args.smac_seed}.txt', 'w',
            buffering=1)
    else:
        data_points, labels = load_subset(args.ds, args.size, args.sampling, args.data_seed)
        performance_log_file = open(
            f'opt_logs/{args.ds}_{method}/log_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}_{args.data_seed}_{args.smac_seed}.txt', 'w',
            buffering=1)
    name = f"{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}_{args.data_seed}"
    incumbent, score, scenario, run_num = run_parameter_estimation(args, args.smac_seed, name)
    parameter_log_file.write(f"{args.smac_seed};{dict(incumbent)};{score};{run_num}\n")
    performance_log_file.close()



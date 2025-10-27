import argparse
import math
import os
import traceback
import warnings
from datetime import datetime
from time import sleep
from timeit import default_timer

from ConfigSpace import Configuration
from pynisher.exceptions import TimeoutException, WallTimeoutException
from smac import Scenario, AlgorithmConfigurationFacade

import numpy as np

from clustering_handler import perform_clustering, eval_clustering_supervised, eval_clustering_unsupervised
from configspace_handler import get_configspace
from data_handler import load_data
from subset_handler import load_subset

global data_points
global label
global smac
global performance_log_file

# from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
import signal
from contextlib import contextmanager

warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)

# strictly enforces time limit, only works on Linux
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise WallTimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    #print("I was here", flush=True)
    try:
        yield
    finally:
        #print("I was here as well", flush=True)
        signal.alarm(0)

# handles a single run
def clustering_runner(config: Configuration, seed: int = 0) -> float:
    config_dict = dict(config)
    score_sum = 0
    metricss = []
    #print(config_dict, flush=True)
    remaining_time = int(math.ceil(smac.optimizer.remaining_walltime))
    print("remaining time:", remaining_time, "seconds")
    try:
        with time_limit(remaining_time):
            for seed in range(3): # take average of 3 seeds for clustering algorithm
                start = default_timer()
                clustering = perform_clustering(data_points, config_dict["method"], config_dict, seed)
                time = float(default_timer() - start)
                metrics = eval_clustering_supervised(clustering, labels)
                if config_dict["ds"] != "large" and config_dict["ds"] != "verylarge" and config_dict["ds"] != "verylarge3":
                    metrics |= eval_clustering_unsupervised(clustering, data_points)
                metrics |= {"clu_num": len(np.unique(clustering)), "clu_histogram": np.unique(clustering, return_counts=True)[1].tolist(), "Time": time}
                if config_dict["supervised"]:
                    cur_score = 2 - metrics["AMI"] - metrics["ARI"]
                else:
                    cur_score = 2 - metrics["SilhouetteScore"] - metrics["DISCO5"]
                score_sum += cur_score
                metricss.append(metrics)
            score = score_sum/5 #should be 3, doesn't change result, since universally applied
    except WallTimeoutException:
        print("Timed out!", flush=True)
        score = np.inf
        raise WallTimeoutException("Timed out!")
    except Exception:
        print(traceback.format_exc(), flush=True)
        score = np.inf
    finally:
        performance_log_file.write(f"{config_dict}, {score}\n")
        for metrics in metricss:
            performance_log_file.write(f"-\t{config_dict}: {metrics}\n")
    return score

# performs search for suitable hyperparameter configuration
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="aggregation", type=str, help='Dataset')
    parser.add_argument('--size', default=1, type=float, help='Size of Dataset')
    parser.add_argument('--sampling', default="random", type=str, help='Downsampling Strategy')
    parser.add_argument('--method', default="spectral", type=str, help='Clustering Method')
    parser.add_argument('--budget', default=30, type=int, help='SMAC AutoML Budget (in seconds)')
    parser.add_argument('--data_seed', default=-1, type=int, help='Seed for dataset (-1 iterates over 3 seeds)')
    parser.add_argument('--smac_seed', default=-1, type=int, help='Seed for SMAC3 (-1 iterates over 3 seeds)')
    parser.add_argument('--supervised', default=1, type=int, help='Use supervised scoring')
    parser.add_argument('--overwrite', default=1, type=int, help='Overwrite existing results')
#    parser.add_argument('--trialbudget', default=36, type=int, help='Hard limit on individual configuration run length (in seconds), if active, enforces hard cutoff at end of full budget')

    args = parser.parse_args()
    args.supervised = args.supervised == 1
    args.overwrite = args.overwrite == 1

    print(args)

    method = args.method
    supervised = args.supervised
    sup_string = "supervised" if supervised else "unsupervised"

    if not os.path.exists("opt_logs"):
        os.makedirs("opt_logs", exist_ok=True)
    if not os.path.exists("param_logs"):
        os.makedirs("param_logs", exist_ok=True)

    if not os.path.exists(f"opt_logs/{args.ds}_{method}"):
        os.makedirs(f"opt_logs/{args.ds}_{method}", exist_ok=True)

    if not os.path.exists(f"param_logs/{args.ds}_{method}"):
        os.makedirs(f"param_logs/{args.ds}_{method}", exist_ok=True)

    if args.smac_seed == -1:
        smac_seeds = range(3)
    else:
        smac_seeds = [args.smac_seed]

    if args.sampling not in ["protras", "dendis", "dides"] and args.size < 1:
        if args.data_seed == -1:
            data_seeds = range(3)
        else:
            data_seeds = [args.data_seed]
    else:
        data_seeds = [0]

    if not args.overwrite:
        if args.size == 1.0:
            if os.path.isfile(f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_full_{args.budget}.csv'):
                with open(
                    f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_full_{args.budget}.csv', 'r') as parameter_log_file:
                    line = parameter_log_file.readline()
                    if line != "":
                        raise FileExistsError
        else:
            if os.path.isfile(f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}.csv'):
                with open(
                    f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}.csv', 'r') as parameter_log_file:
                    line = parameter_log_file.readline()
                    if line != "":
                        raise FileExistsError


    if args.size == 1.0:
        parameter_log_file = open(
            f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_full_{args.budget}.csv', 'w',
            buffering=1)
    else:
        parameter_log_file = open(
            f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}.csv',
            'w',
            buffering=1)

    for args.data_seed in data_seeds: #x3
        print("data_seed", args.data_seed, flush=True)
        if args.size == 1.0:
            data_points, labels = load_data(args.ds)
        else:
            data_points, labels = load_subset(args.ds, args.size, args.sampling, args.data_seed)
        for args.smac_seed in smac_seeds: #x3
            smac = None
            print("smac_seed", args.smac_seed, flush=True)
            if args.size == 1.0:
                performance_log_name = f'opt_logs/{args.ds}_{method}/log_{args.ds}_{method}_{sup_string}_full_{args.budget}_{args.smac_seed}.txt'
            else:
                performance_log_name = f'opt_logs/{args.ds}_{method}/log_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}_{args.data_seed}_{args.smac_seed}.txt'
            performance_log_file = open(performance_log_name, 'w', buffering=1)
            name = f"{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}_{args.data_seed}"
            scenario = Scenario(get_configspace(args.method, args.ds, args), use_default_config=True,
                                n_trials=10000000, walltime_limit=args.budget, seed=args.smac_seed, name=name, deterministic=True)
            # uses AlgorithmConfigurationFacade as it starts with default values, HPO is special case of AC, so appropriate. Initially used HPO Facade, but early tests showed insufficient exploration for clustering (e.g., AC consistently found best DBSCAN configuration on Complex-9, while HPO Facade did not)
            smac = AlgorithmConfigurationFacade(scenario, clustering_runner, overwrite=True)
            incumbent = smac.optimize()
            score = smac.runhistory.get_min_cost(incumbent)
            run_num = len(smac.runhistory.items())
            print(incumbent, score, run_num, flush=True)
            parameter_log_file.write(f"{args.smac_seed} {args.data_seed};{dict(incumbent)};{score};{run_num}\n")
            performance_log_file.close()

    parameter_log_file.close()



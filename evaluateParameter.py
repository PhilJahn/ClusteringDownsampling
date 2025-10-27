import argparse
import ast
import os
import traceback
from timeit import default_timer
import numpy as np

from clustering_handler import perform_clustering, eval_clustering_supervised, eval_clustering_unsupervised
from data_handler import load_data
from subset_handler import load_subset

# runs clustering with a given configuration
def clustering_runner(config, data_points, labels, seed: int = 0):
    config_dict = dict(config)
    start = default_timer()
    clustering = perform_clustering(data_points, method, config_dict, seed)
    time = float(default_timer() - start)
    metrics = eval_clustering_supervised(clustering, labels)
    if len(data_points) < 60000:
        metrics |= eval_clustering_unsupervised(clustering, data_points)
    metrics |= {"clu_num": len(np.unique(clustering)), "clu_histogram": np.unique(clustering, return_counts=True)[1].tolist(), "Time": time}
    metrics["sup_score"] = 2 - metrics["AMI"] - metrics["ARI"]
    if len(data_points) < 60000:
        metrics["unsup_score"] = 2 - metrics["SilhouetteScore"] - metrics["DISCO5"]
    return metrics

# processes hyperparameter values based on scaling function
# prior knowledge of scaling functions baked in to make examinations easier (i.e. only applies scaling to hyperparameters wherescaling behavior was found)
def param_process(params, scaling, size, evalsize):
    scale = evalsize/size
    if scaling == "sample_mult":
        if "min_samples" in params.keys():
            params["min_samples"] = round(params["min_samples"]*scale)
        if "n_neighbors" in params.keys():
            params["n_neighbors"] = round(params["n_neighbors"]*scale)
        if "min_cluster_size" in params.keys():
            params["min_cluster_size"] = round(params["min_cluster_size"]*scale)
    elif scaling == "root_unsup":
        if "min_samples" in params.keys():
            params["min_samples"] = round(params["min_samples"]*scale)
        if "n_neighbors" in params.keys():
            params["n_neighbors"] = round(params["n_neighbors"]*scale)
        if "min_cluster_size" in params.keys():
            params["min_cluster_size"] = round(params["min_cluster_size"]*scale)
        if "n_clusters" in params.keys():
            params["n_clusters"] = round(params["n_clusters"]*(scale**0.5))
        if "n_components" in params.keys():
            params["n_components"] = round(params["n_components"]*(scale**0.5))
    elif scaling == "sample_square":
        if "min_samples" in params.keys():
            params["min_samples"] = round(params["min_samples"]*scale*scale)
        if "n_neighbors" in params.keys():
            params["n_neighbors"] = round(params["n_neighbors"]*scale*scale)
        if "min_cluster_size" in params.keys():
            params["min_cluster_size"] = round(params["min_cluster_size"]*scale*scale)
    elif scaling == "sample_root":
        if "min_samples" in params.keys():
            params["min_samples"] = round(params["min_samples"]*(scale**0.5))
        if "n_neighbors" in params.keys():
            params["n_neighbors"] = round(params["n_neighbors"]*(scale**0.5))
        if "min_cluster_size" in params.keys():
            params["min_cluster_size"] = round(params["min_cluster_size"]*(scale**0.5))
    return params


# evaluates hyperparameter configurations on different (or same) dataset size
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="aggregation", type=str, help='Dataset')
    parser.add_argument('--size', default=0.75, type=float, help='Size of Dataset used for parameters')
    parser.add_argument('--evalsize', default=1.0, type=float, help='Size of Dataset used for evaluation')
    parser.add_argument('--sampling', default="kcentroid", type=str, help='Downsampling Strategy for parameters')
    parser.add_argument('--evalsampling', default="kcentroid", type=str, help='Downsampling Strategy for evaluation')
    parser.add_argument('--method', default="hdbscan", type=str, help='Clustering Method')
    parser.add_argument('--budget', default=3600, type=int, help='SMAC AutoML Budget (in seconds)')
    parser.add_argument('--supervised', default=1, type=int, help='Use supervised scoring')
    #options: no_scaling, sample_mult
    parser.add_argument('--scaling', default="sample_mult", type=str, help='Scaling Strategy')
    args = parser.parse_args()
    args.supervised = args.supervised == 1

    method = args.method
    supervised = args.supervised
    sup_string = "supervised" if supervised else "unsupervised"

    if not os.path.exists("eval_logs"):
        os.makedirs("eval_logs", exist_ok=True)
    if not os.path.exists(f"eval_logs/{args.ds}_{method}"):
        os.makedirs(f"eval_logs/{args.ds}_{method}", exist_ok=True)

    if args.size == 1.0:
        param_file_path = f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_full_{args.budget}.csv'
    else:
        param_file_path = f'param_logs/{args.ds}_{method}/params_{args.ds}_{method}_{sup_string}_{args.sampling}_{args.size}_{args.budget}.csv'

    params_store = []
    score_store = []
    count_store = []
    if os.path.exists(param_file_path):
        param_file = open(param_file_path, 'r')
        try:
            line = param_file.readline()
            while line != '':
                line_split = line.split(';')
                # https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
                params = ast.literal_eval(line_split[1])
                params_processed = param_process(params, args.scaling, args.size, args.evalsize)
                params_store.append(params_processed)
                score_store.append(float(line_split[2]))
                count_store.append(int(line_split[3]))
                line = param_file.readline()
        except Exception as e:
            print("Error while reading params")
            raise e

        if args.evalsize == 1.0:
            data_seeds = [0]
        else:
            data_seeds = range(5)

        eval_path = f'eval_logs/{args.ds}_{method}/log_{args.ds}_{method}_{sup_string}_from_{args.sampling}_{args.size}_on_{args.evalsampling}_{args.evalsize}_{args.budget}_{args.scaling}.txt'
        #if os.path.exists(eval_path):
        #    raise Exception

        eval_log_file = open(eval_path, 'w', buffering=1)

        mean_score_store = []
        for data_seed in data_seeds:
            if args.evalsize == 1.0:
                data_points, labels = load_data(args.ds)
            else:
                data_points, labels = load_subset(args.ds, args.size, args.evalsampling, data_seed)
            i = 0
            for params in params_store:
                try:
                    score_full_store = []
                    metrics_store = []
                    if method in ["kmeans", "em", "spectral"]:
                        seeds = range(5)
                    else:
                        seeds = [0]
                    for seed in seeds:
                        metrics = clustering_runner(params, data_points, labels)
                        score_full = metrics["sup_score"] if supervised else metrics["unsup_score"]
                        score_full_store.append(score_full)
                        metrics_store.append(metrics)
                    score_sub = score_store[i]
                    count_sub = count_store[i]
                    eval_log_file.write(f"{data_seed} {params} ({score_sub} -> {np.mean(score_full_store)} +/- {np.std(score_full_store)}) {count_sub}\n")
                    mean_score_store.append(np.mean(score_full_store))
                    for seed in seeds:
                        metric_string = f"\t{seed} {data_seed} {params} ({score_sub} -> {score_full_store[seed]}): {metrics_store[seed]}\n"
                        eval_log_file.write(metric_string)
                except ValueError:
                    score_sub = score_store[i]
                    count_sub = count_store[i]
                    score_full_store = [4]
                    mean_score_store.append(np.mean(score_full_store))
                    eval_log_file.write(f"{data_seed} {params} ({score_sub} -> {np.mean(score_full_store)} +/- {np.std(score_full_store)}) {count_sub}\n")
                    for seed in range(5):
                        metric_string = f"\t{seed} {data_seed} {params} ({score_sub} -> {score_full_store[0]}): ValueError\n"
                        eval_log_file.write(metric_string)
                except Exception:
                    print(traceback.format_exc())
                    raise Exception

                i+=1
        eval_log_file.write(f"full: {np.mean(score_store)} -> {np.mean(mean_score_store)} +/- {np.std(mean_score_store)} {np.mean(count_store)}\n")
        eval_log_file.close()
    else:
        print(f"Parameters for {param_file_path} are not yet determined")



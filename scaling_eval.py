import argparse
import ast
import os
from pprint import pprint

import numpy as np
import scipy
from sortedcontainers import SortedSet

from similarity.imbalance_degree import imbalance_degree_histogram


def file_parser(method, files, path, name):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if method == "dbscan":
        params = ["eps", "min_samples"]
    elif method == "kmeans":
        params = ["n_clusters", "init"]
    elif method == "spectral_gamma":
        params = ["n_clusters", "affinity", "gamma", "n_neighbors", "assign_labels"]
    elif method == "spectral_nn":
        params = ["n_clusters", "affinity", "gamma", "n_neighbors", "assign_labels"]
    elif method == "agglomerative":
        params = ["n_clusters", "connectivity", "n_neighbors", "linkage"]
    elif method == "em":
        params = ["n_components", "init_params", "covariance_type"]
    else:
        raise ValueError("Unknown method")

    sup_dict = {}
    unsup_dict = {}
    sil_dict = {}
    disco5_dict = {}
    disco10_dict = {}
    clunum_dict = {}
    imb_dict = {}
    ctr_dict = {}
    hist_dict = {}

    all_scores_sup = []
    all_scores_unsup = []

    value_dict = {}
    for param in params:
        value_dict[param] = set()

    for file_path in files:
        if os.path.exists(file_path):
            file = open(file_path, 'r')
            line = file.readline()
            while line != '':
                line = line.strip("\n")
                line_split = line.split(';')

                config = line_split[5]
                config_dict = ast.literal_eval(config)
                key = ""
                for param in params:
                    if key != "":
                        key += "_"
                    if config_dict[param] == "random_from_data":
                        config_dict[param] = "randomfromdata"
                    if config_dict[param] == "kneighbors_graph":
                        config_dict[param] = "kneighborsgraph"
                    if config_dict[param] == "nearest_neighbors":
                        config_dict[param] = "nearestneighbors"
                    if config_dict[param] == "cluster_qr":
                        config_dict[param] = "clusterqr"
                    key += str(config_dict[param])
                    value_dict[param].add(config_dict[param])
                if key in ctr_dict.keys():
                    ctr_dict[key] += 1
                else:
                    ctr_dict[key] = 1
                    sup_dict[key] = []
                    unsup_dict[key] = []
                    sil_dict[key] = []
                    disco5_dict[key] = []
                    disco10_dict[key] = []
                    clunum_dict[key] = []
                    imb_dict[key] = []
                    hist_dict[key] = []

                unsup_score = float(line_split[-4])
                unsup_dict[key].append(unsup_score)
                sup_score = float(line_split[-5])
                sup_dict[key].append(sup_score)

                scores = line_split[6]
                if scores != "crashed":
                    score_dict = ast.literal_eval(scores)
                    sil_dict[key].append(score_dict["SilhouetteScore"])
                    disco5_dict[key].append(score_dict["DISCO5"])
                    disco10_dict[key].append(score_dict["DISCO10"])
                    clunum_dict[key].append(score_dict["clu_num"])
                    imb_dict[key].append(imbalance_degree_histogram(score_dict["clu_histogram"])[1])
                    hist_dict[key].append(score_dict["clu_histogram"])

                else:
                    sup_dict[key].append(-2)
                    unsup_dict[key].append(-2)
                    sil_dict[key].append(-1)
                    disco5_dict[key].append(-1)
                    disco10_dict[key].append(-1)
                    clunum_dict[key].append(-1)
                    imb_dict[key].append(-1)
                    hist_dict[key].append(-1)
                line = file.readline()
            file.close()
        else:
            print("File not found", file_path)#
    np.save(f"{path}/{name}_sup_dict.npy", sup_dict)
    np.save(f"{path}/{name}_unsup_dict.npy", unsup_dict)
    np.save(f"{path}/{name}_sil_dict.npy", sil_dict)
    np.save(f"{path}/{name}_disco5_dict.npy", disco5_dict)
    np.save(f"{path}/{name}_disco10_dict.npy", disco10_dict)
    np.save(f"{path}/{name}_clunum_dict.npy", clunum_dict)
    np.save(f"{path}/{name}_imb_dict.npy", imb_dict)
    np.save(f"{path}/{name}_hist_dict.npy", hist_dict)

    for key in sup_dict.keys():
        all_scores_sup.append(np.mean(sup_dict[key]))
        all_scores_unsup.append(np.mean(unsup_dict[key]))

    all_scores_sup_sorted = np.flip(np.sort(all_scores_sup))
    all_scores_unsup_sorted = np.flip(np.sort(all_scores_unsup))

    rank_sup_dict = {}
    rank_unsup_dict = {}

    for key in sup_dict.keys():
        unsup_score = np.mean(unsup_dict[key])
        sup_score = np.mean(sup_dict[key])
        rank_unsup = np.where(all_scores_unsup_sorted == unsup_score)[0]  # + 1
        rank_unsup_dict[key] = min(rank_unsup) + 1

        rank_sup = np.where(all_scores_sup_sorted == sup_score)[0]  # + 1
        rank_sup_dict[key] = min(rank_sup) + 1

    np.save(f"{path}/{name}_sup_rank_dict.npy", rank_sup_dict)
    np.save(f"{path}/{name}_unsup_rank_dict.npy", rank_unsup_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--up_ds', default="scaling2", type=str, help='Upsampled Dataset')
    parser.add_argument('--reg_ds', default="scaling1", type=str, help='Regular Dataset')
    parser.add_argument('--factor', default=2.0, type=float, help='Upsampling Ratio (always 2 for paper)')
    parser.add_argument('--metric', default="unsup", type=str, help='Which metric to use for the distance (sup = supervised, unsup = unsupervised, also: sil, disco5, disco10, sup_rank, unsup_rank, imb, clu_num)')
    parser.add_argument('--rebuild', default=0, type=int, help='Rebuild log files (boolean)')
    parser.add_argument('--method', default="agglomerative", type=str, help='Clustering Method')
    parser.add_argument('--submethod', default="none", type=str, help='Subselection of method (only used for Agglomerative Linkage types)')



    args = parser.parse_args()
    print(args)

    up_data = args.up_ds
    reg_data = args.reg_ds
    factor = args.factor
    metric = args.metric
    rebuild_files = args.rebuild == 1
    method = args.method
    submethod = args.submethod

    if rebuild_files:
        for ds in [up_data, reg_data]:
            kmeans_files = [f"grid_evals/{ds}_random_1.0_kmeans_n_clusters_init_none_100.txt"]
            file_parser("kmeans", kmeans_files, "grid_dicts", f"{ds}_kmeans")

            dbscan_files = [f"grid_evals/{ds}_random_1.0_dbscan_eps_min_samples_none_100.txt"]
            file_parser("dbscan", dbscan_files, "grid_dicts", f"{ds}_dbscan")

            em_files = [f"grid_evals/{ds}_random_1.0_em_n_components_init_params_covariance_type_100.txt"]
            file_parser("em", em_files, "grid_dicts", f"{ds}_em")

            spectral_files = [f"grid_evals/{ds}_random_1.0_spectral_n_clusters_gamma_none_100.txt", f"grid_evals/{ds}_random_1.0_spectral_n_clusters_gamma_assign_labels_c_100.txt", f"grid_evals/{ds}_random_1.0_spectral_n_clusters_gamma_assign_labels_d_100.txt"]
            file_parser("spectral_gamma", spectral_files, "grid_dicts", f"{ds}_spectral_gamma")

            spectral_files = [f"grid_evals/{ds}_random_1.0_spectral_n_clusters_n_neighbors_none_100.txt", f"grid_evals/{ds}_random_1.0_spectral_n_clusters_n_neighbors_assign_labels_c_100.txt", f"grid_evals/{ds}_random_1.0_spectral_n_clusters_n_neighbors_assign_labels_d_100.txt"]
            file_parser("spectral_nn", spectral_files, "grid_dicts", f"{ds}_spectral_nn")

            agg_files = [f"grid_evals/{ds}_random_1.0_agglomerative_n_clusters_n_neighbors_linkage_100.txt"]
            file_parser("agglomerative", agg_files, "grid_dicts", f"{ds}_agglomerative")

            agg_files = [f"grid_evals/{ds}_random_1.0_agglomerative_n_clusters_linkage_none_100.txt"]
            file_parser("agglomerative", agg_files, "grid_dicts", f"{ds}_agglomerative_unstructured")

    if method in ["agglomerative", "agglomerative_unstructured"] and submethod != "none":
        print(method, submethod, metric)
    else:
        print(method, metric)


    up_score = np.load(f"grid_dicts/{up_data}_{method}_{metric}_dict.npy", allow_pickle=True).item()
    reg_score = np.load(f"grid_dicts/{reg_data}_{method}_{metric}_dict.npy", allow_pickle=True).item()

    params = []
    if method == "dbscan":
        params = ["eps", "min_samples"]
        scaling_options = [{"eps": "none", "min_samples": "none"}, {"eps": "none", "min_samples": "linear_int"}, {"eps": "linear_float", "min_samples": "none"}, {"eps": "linear_float", "min_samples": "linear_int"},
                           {"eps": "none", "min_samples": "square_int"}, {"eps": "square_float", "min_samples": "none"}, {"eps": "square_float", "min_samples": "square_int"},
                           {"eps": "none", "min_samples": "root_int"}, {"eps": "root_float", "min_samples": "none"}, {"eps": "root_float", "min_samples": "root_int"},
                           {"eps": "linear_float", "min_samples": "root_int"}, {"eps": "square_float", "min_samples": "root_int"},
                           {"eps": "linear_float", "min_samples": "square_int"}, {"eps": "root_float", "min_samples": "square_int"}
                           ]
    elif method == "kmeans":
        params = ["n_clusters", "init"]
        scaling_options = [{"n_clusters": "none", "init": "none"}, {"n_clusters": "linear_int", "init": "none"}, {"n_clusters": "root_int", "init": "none"}, {"n_clusters": "square_int", "init": "none"}]
    elif method == "spectral_gamma":
        params = ["n_clusters", "affinity", "gamma", "n_neighbors", "assign_labels"]
        scaling_options = [{"n_clusters": "none", "affinity": "none", "gamma": "none", "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "none", "n_neighbors": "none",
                            "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "none", "n_neighbors": "none",
                            "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "none", "n_neighbors": "none",
                            "assign_labels": "none"},

                           {"n_clusters": "none", "affinity": "none", "gamma": "linear_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "linear_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "linear_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "linear_float",
                            "n_neighbors": "none", "assign_labels": "none"},

                           {"n_clusters": "none", "affinity": "none", "gamma": "root_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "root_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "root_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "root_float",
                            "n_neighbors": "none", "assign_labels": "none"},

                           {"n_clusters": "none", "affinity": "none", "gamma": "square_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "square_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "square_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "square_float",
                            "n_neighbors": "none", "assign_labels": "none"},
                           ]
    elif method == "spectral_nn":
        params = ["n_clusters", "affinity", "gamma", "n_neighbors", "assign_labels"]
        scaling_options = [{"n_clusters": "none", "affinity": "none", "gamma": "none", "n_neighbors": "none", "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "none", "n_neighbors": "none",
                            "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "none", "n_neighbors": "none",
                            "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "none", "n_neighbors": "none",
                            "assign_labels": "none"},

                           {"n_clusters": "none", "affinity": "none", "gamma": "none", "n_neighbors": "linear_int",
                            "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "none", "n_neighbors": "linear_int",
                            "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "none", "n_neighbors": "linear_int",
                            "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "none", "n_neighbors": "linear_int",
                            "assign_labels": "none"},

                           {"n_clusters": "none", "affinity": "none", "gamma": "none", "n_neighbors": "root_int",
                            "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "none", "n_neighbors": "root_int",
                            "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "none", "n_neighbors": "root_int",
                            "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "none", "n_neighbors": "root_int",
                            "assign_labels": "none"},

                           {"n_clusters": "none", "affinity": "none", "gamma": "none", "n_neighbors": "square_int",
                            "assign_labels": "none"},
                           {"n_clusters": "linear_int", "affinity": "none", "gamma": "none", "n_neighbors": "square_int",
                            "assign_labels": "none"},
                           {"n_clusters": "root_int", "affinity": "none", "gamma": "none", "n_neighbors": "square_int",
                            "assign_labels": "none"},
                           {"n_clusters": "square_int", "affinity": "none", "gamma": "none", "n_neighbors": "square_int",
                            "assign_labels": "none"},
                           ]

    elif method == "agglomerative":
        params = ["n_clusters", "connectivity", "n_neighbors", "linkage"]
        scaling_options = [{"n_clusters": "none","connectivity": "none","n_neighbors": "none","linkage": "none"},
                           {"n_clusters": "linear_int","connectivity": "none","n_neighbors": "none","linkage": "none"},
                           {"n_clusters": "root_int","connectivity": "none","n_neighbors": "none","linkage": "none"},
                           {"n_clusters": "square_int","connectivity": "none","n_neighbors": "none","linkage": "none"},

                           {"n_clusters": "none", "connectivity": "none", "n_neighbors": "linear_int", "linkage": "none"},
                           {"n_clusters": "linear_int", "connectivity": "none", "n_neighbors": "linear_int",
                            "linkage": "none"},
                           {"n_clusters": "root_int", "connectivity": "none", "n_neighbors": "linear_int", "linkage": "none"},
                           {"n_clusters": "square_int", "connectivity": "none", "n_neighbors": "linear_int",
                            "linkage": "none"},

                           {"n_clusters": "none", "connectivity": "none", "n_neighbors": "root_int",
                            "linkage": "none"},
                           {"n_clusters": "linear_int", "connectivity": "none", "n_neighbors": "root_int",
                            "linkage": "none"},
                           {"n_clusters": "root_int", "connectivity": "none", "n_neighbors": "root_int",
                            "linkage": "none"},
                           {"n_clusters": "square_int", "connectivity": "none", "n_neighbors": "root_int",
                            "linkage": "none"},

                           {"n_clusters": "none", "connectivity": "none", "n_neighbors": "square_int",
                            "linkage": "none"},
                           {"n_clusters": "linear_int", "connectivity": "none", "n_neighbors": "square_int",
                            "linkage": "none"},
                           {"n_clusters": "root_int", "connectivity": "none", "n_neighbors": "square_int",
                            "linkage": "none"},
                           {"n_clusters": "square_int", "connectivity": "none", "n_neighbors": "square_int",
                            "linkage": "none"},
                           ]

    elif method == "agglomerative_unstructured":
        params = ["n_clusters", "connectivity", "n_neighbors", "linkage"]
        scaling_options = [{"n_clusters": "none","connectivity": "none","n_neighbors": "none","linkage": "none"},
                           {"n_clusters": "linear_int","connectivity": "none","n_neighbors": "none","linkage": "none"},
                           {"n_clusters": "root_int","connectivity": "none","n_neighbors": "none","linkage": "none"},
                           {"n_clusters": "square_int","connectivity": "none","n_neighbors": "none","linkage": "none"},
                           ]

    elif method == "em":
        params = ["n_components", "init_params", "covariance_type"]
        scaling_options = [{"n_components": "none", "init_params": "none",  "covariance_type": "none"}, {"n_components": "linear_int", "init_params": "none",  "covariance_type": "none"},
                           {"n_components": "root_int", "init_params": "none", "covariance_type": "none"}, {"n_components": "square_int", "init_params": "none", "covariance_type": "none"}
                           ]


    param_choices = {}
    for key in reg_score.keys():
        param_vals = key.split("_")
        for i in range(len(param_vals)):
            if params[i] in param_choices.keys():
                param_choices[params[i]].add(param_vals[i])
            else:
                param_choices[params[i]] = SortedSet([param_vals[i]])
    if submethod == "ward":
        param_choices["linkage"] = SortedSet(["ward"])
    elif submethod == "complete":
        param_choices["linkage"] = SortedSet(["complete"])
    elif submethod == "average":
        param_choices["linkage"] = SortedSet(["average"])
    elif submethod == "single":
        param_choices["linkage"] = SortedSet(["single"])
    elif submethod == "other":
        param_choices["linkage"] = SortedSet(["complete", "average", "single"])

    #print(param_choices)


    #print(param_choices)
    best_scaling = None
    min_score = np.inf
    best_std = -1
    best_diff = -1

    if not os.path.exists("scale_logs"):
        os.makedirs("scale_logs", exist_ok=True)
    scaling_eval_path = f'scale_logs/log_{args.reg_ds}_to_{args.up_ds}_{args.method}_{args.submethod}_{args.metric}.txt'
    eval_log_file = open(scaling_eval_path, 'w', buffering=1)

    for scaling in scaling_options:
        diff = []
        up_performances = []
        reg_performances = []
        up_keys = []
        reg_keys = []
        up_ctr = 0
        reg_ctr = 0
        same_ctr = 0
        for key in reg_score.keys():
            reg_val = reg_score[key]
            param_vals = key.split("_")
            up_key = ""
            found = True
            if (method == "agglomerative" or method == "agglomerative_unstructured") and submethod != "none" and param_vals[3] not in param_choices["linkage"]:
                continue
            for i in range(len(param_vals)):
                scaling_behavior = scaling[params[i]]
                if up_key != "":
                    up_key += "_"
                if scaling_behavior == "none":
                    up_key += str(param_vals[i])
                else:
                    if "linear" in scaling_behavior:
                        up_value = factor * float(param_vals[i])
                    elif "square" in scaling_behavior:
                        up_value = factor * factor * float(param_vals[i])
                    elif "root" in scaling_behavior:
                        up_value = (factor**0.5) * float(param_vals[i])
                        if "int" in scaling_behavior:
                            up_value = round(up_value)
                    else:
                        raise NotImplementedError
                    #print(param_choices[params[i]])
                    if "float" in scaling_behavior:
                        #https://stackoverflow.com/questions/8466014/how-to-convert-a-python-set-to-a-numpy-array
                        options = np.array(list(param_choices[params[i]]), dtype=float)
                    else:
                        options = np.array(list(param_choices[params[i]]), dtype=int)
                    options_diff = np.abs(np.array(options) - up_value)
                    arg_min_diff = np.argmin(options_diff)
                    if options_diff[arg_min_diff] < 0.01:
                        up_key += str(options[arg_min_diff])
                        #print(param_vals[i], ":", up_value, "->", options[arg_min_diff])
                    else:
                        #print(param_vals[i], ":", up_value, "-> no option found")
                        found = False
                        break
            if found:
                if up_key in up_score.keys():
                    diff.append(abs(np.mean(reg_score[key]) - np.mean(up_score[up_key])))
                    up_performances.append(np.mean(up_score[up_key]))
                    reg_performances.append(np.mean(reg_score[key]))
                    up_keys.append(up_key)
                    reg_keys.append(key)
                    if np.mean(reg_score[key]) > np.mean(up_score[up_key]):
                        reg_ctr += 1
                    elif np.mean(reg_score[key]) < np.mean(up_score[up_key]):
                        up_ctr += 1
                    else:
                        same_ctr += 1
                #print(key, up_key, "->", np.mean(reg_score[key]), np.mean(up_score[up_key]), abs(np.mean(reg_score[key]) - np.mean(up_score[up_key])))

        # print(up_performances)
        # print(reg_performances)
        # rank_one_retention = 0
        # rank_ones = 0
        # for i in range(len(up_keys)):
        #     up_key = up_keys[i]
        #     reg_key = reg_keys[i]
        #     mean_up_score = np.mean(up_score[up_key])
        #     mean_reg_score = np.mean(reg_score[reg_key])
        #     if mean_reg_score == 1:
        #         print("rank one included", mean_up_score, reg_key, up_key)
        #         rank_ones += 1
        #         if mean_up_score == 1:
        #             rank_one_retention += 1
        # if rank_ones >0:
        #     print("rank one retenetion", rank_one_retention/rank_ones)
        # else:
        #     print("no rank one")

        # up_performances_sorted = np.flip(np.sort(up_performances))
        # reg_performances_sorted = np.flip(np.sort(reg_performances))
        #
        # rank_sup_dict = {}
        # rank_unsup_dict = {}
        #
        # rank_diffs = []
        # up_ranking = []
        # for i in range(len(up_keys)):
        #     up_key = up_keys[i]
        #     reg_key = reg_keys[i]
        #     mean_up_score = np.mean(up_score[up_key])
        #     mean_reg_score = np.mean(reg_score[reg_key])
        #
        #     up_rank = min(np.where(up_performances_sorted == mean_up_score)[0]) + 1
        #     reg_rank = min(np.where(reg_performances_sorted == mean_reg_score)[0]) + 1
        #     rank_diff = abs(reg_rank - up_rank)
        #     if rank_diff > 1000:
        #         print()
        #     rank_diffs.append(rank_diff)




        score = np.mean(diff)+np.std(diff)
        if score < min_score:
            min_score = score
            best_scaling = scaling
            best_diff = np.mean(diff)
            best_std = np.std(diff)

        median_score = np.median(diff)



        eval_log_file.write(f"{scaling}, diff: {np.mean(diff):.3f}, std: {np.std(diff):.3f}, tested: {len(diff)}")
        eval_log_file.write(f"up: {np.mean(up_performances):.3f}, reg: {np.mean(reg_performances):.3f}, ")
        eval_log_file.write(f"up+: {up_ctr/len(diff):.3f}, reg+: {reg_ctr / len(diff):.3f}, same: {same_ctr / len(diff):.3f}, ")
        eval_log_file.write(f"score: {np.mean(diff) + np.std(diff):.3f}, median_score: {median_score:.3f}\n")
    eval_log_file.write(f"{best_scaling} ${best_diff:.3f} \\pm {best_std:.3f}$ {min_score:.3f}")
    eval_log_file.close()




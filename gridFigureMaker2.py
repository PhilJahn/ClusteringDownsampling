import os
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from scaling_eval import file_parser

if __name__ == "__main__":
    if not os.path.exists("grid_figures"):
        os.makedirs("grid_figures")
    plt.rcParams.update({'font.size': 8})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    fig_text = {}
    fig_text[0.01] = "Random - 1%"
    fig_text[0.1] = "Random - 10%"
    fig_text[0.25] = "Random - 25%"
    fig_text[0.5] = "Random - 50%"
    fig_text[0.75] = "Random - 75%"
    fig_text[1.0] = "Full Dataset"

    method = "dbscan"
    metric = "sup"
    ds = "complex9"
    yaxis = 1
    xaxis = 0
    scales = [0.01, 0.1, 0.5, 1.0]

    params = []
    defaults = {}
    max_val = {}
    if method == "dbscan":
        params = ["eps", "min_samples"]
        file_name = "eps_min_samples_none"
        defaults = {"eps": "0.5", "min_samples": "5"}
        max_val = {"eps": 0.48, "min_samples": 50}
        size = (3.5, 2.8)
    elif method == "kmeans":
        params = ["n_clusters", "init"]
        file_name = "n_clusters_init_none"
        defaults = {"n_clusters": "9", "init": "k-means++"}
        max_val = {"n_clusters":30}
        size = (3.5, 2.5)
    elif method == "spectral_nn":
        params = ["n_clusters", "affinity", "gamma", "n_neighbors", "assign_labels"]
        defaults = {"n_clusters": "9", "affinity": "nearestneighbors", "gamma": "1.0", "n_neighbors": "10",
                    "assign_labels": "kmeans"}
    elif method == "spectral_gamma":
        params = ["n_clusters", "affinity", "gamma", "n_neighbors", "assign_labels"]
        defaults = {"n_clusters": "9", "affinity": "rbf", "gamma": "1.0", "n_neighbors": "10",
                    "assign_labels": "kmeans"}
    elif method == "agglomerative":
        params = ["n_clusters", "connectivity", "n_neighbors", "linkage"]
        defaults = {"n_clusters": "9", "connectivity": "kneighborsgraph", "n_neighbors": "10", "linkage": "ward"}
        file_name = "n_clusters_n_neighbors_linkage"
        max_val = {"n_clusters":40, "n_neighbors":10}
        size = (3.5, 6)
    elif method == "em":
        params = ["n_components", "init_params", "covariance_type"]
        defaults = {"n_components": "9", "init_params": "kmeans", "covariance_type": "full"}
        file_name = "n_components_init_params_covariance_type"
        max_val = {"n_components":30}
        size = (3.5, 2.5)
    for scale in scales:
        if not os.path.exists(f"grid_dicts/{ds}_{scale}_{method}_{metric}_{file_name}_dict.npy"):
            files = [f"grid_evals/{ds}_random_{scale}_{method}_{file_name}_100.txt"]
            file_parser(method, files, "grid_dicts", f"{ds}_{scale}_{method}_{file_name}")

    val24 = 0.024 / max_val["eps"]
    val56 = 0.056 / max_val["eps"]
    val49 = 0.049 / max_val["eps"]
    val39 = 0.039 / max_val["eps"]
    val36 = 0.036 / max_val["eps"]
    val35 = 0.035 / max_val["eps"]
    val34 = 0.034 / max_val["eps"]
    val63 = 0.063 / max_val["eps"]
    val62 = 0.062 / max_val["eps"]
    val59 = 0.059 / max_val["eps"]
    val82 = 0.082 / max_val["eps"]
    val80 = 0.080 / max_val["eps"]

    val114 = 0.114 / max_val["eps"]
    val116 = 0.116 / max_val["eps"]
    val115 = 0.115 / max_val["eps"]
    val233 = 0.233 / max_val["eps"]
    val234 = 0.234 / max_val["eps"]
    val199 = 0.199 / max_val["eps"]
    val207 = 0.207 / max_val["eps"]
    val218 = 0.218 / max_val["eps"]

    opts = {0.01: np.array([[val114, 1], [val116, 1], [val115, 1], [val233, 6], [val234, 6], [val199,1], [val207,2], [val218, 1]]),
            0.1: np.array([[val63,1], [val63,2], [val62, 4], [val59,2], [val82, 8], [val80, 8]]),
            0.5: np.array([[val39, 5], [val35, 1], [val36,2], [val39, 7], [val35, 2]]),
            1.0: np.array([[]])
            }
    scale_color = {0.01: "red", 0.1: "orange", 0.5: "yellow", 1.0: "white"}
    scale_opts = {0.01: np.array([[]]),
                  0.1: np.array([[val63,10], [val63,20], [val59,20], [val62, 40]]),
                  0.5: np.array([[val39, 10], [val35, 2], [val36,4], [val39, 14], [val35, 4]]),
                  1.0: np.array([[val24, 2], [val56, 49], [val49, 38]]),
                  }
    symbol = {0.01: "X", 0.1: "P", 0.5: "*", 1.0: "."}

    fig, ax = plt.subplots(2,2, figsize=size)

    for s in range(len(scales)):
        scale = scales[s]
        scores = np.load(f"grid_dicts/{ds}_{scale}_{method}_{file_name}_{metric}_dict.npy", allow_pickle=True).item()
        score_keys = list(scores.keys())
        param_vals = {}
        for score_key in score_keys:
            score_key_vals = score_key.split("_")
            for i in range(len(score_key_vals)):
                score_key_val = score_key_vals[i]
                param = params[i]
                include = True
                if param in max_val.keys():
                    if float(score_key_val) > max_val[param]:
                        include = False
                if include:
                    if param not in param_vals:
                        param_vals[param] = []
                    if score_key_val not in param_vals[param]:
                        param_vals[param].append(score_key_val)
        content = np.zeros((len(param_vals[params[yaxis]]), len(param_vals[params[xaxis]])))
        for i in range(len(param_vals[params[xaxis]])):
            for j in range(len(param_vals[params[yaxis]])):
                x_val = param_vals[params[xaxis]][i]
                y_val = param_vals[params[yaxis]][j]
                key_vals = []
                for p in params:
                    key_vals.append(defaults[p])
                key_vals[xaxis] = x_val
                key_vals[yaxis] = y_val
                key = ""
                for key_val in key_vals:
                    if key != "":
                        key += "_"
                    key += f"{key_val}"
                content[j, i] = np.mean(scores[key])
        #print(content)

        xvals = param_vals[params[xaxis]]
        if len(xvals) > 10:
            x_indices = np.linspace(0, len(xvals), endpoint=False, num=10, dtype=int)
            x_labels = np.array(xvals)[x_indices]
            if not x_labels[0].isnumeric() and x_labels[0][-1].isnumeric():
                x_labels = ['%.3f' % float(x_label) for x_label in x_labels]
        else:
            x_indices = range(len(xvals))
            x_labels = xvals
            if not x_labels[0].isnumeric() and x_labels[0][-1].isnumeric():
                x_labels = ['%.3f' % float(x_label) for x_label in x_labels]
        yvals = param_vals[params[yaxis]]
        if len(yvals) > 7:
            y_indices = np.linspace(0, len(yvals), endpoint=False, num=7, dtype=int)
            y_labels = np.array(yvals)[y_indices]
            if not y_labels[0].isnumeric() and y_labels[0][-1].isnumeric():
                y_labels = ['%.3f' % float(y_label) for y_label in y_labels]
        else:
            y_indices = range(len(yvals))
            y_labels = yvals
            if not y_labels[0].isnumeric() and y_labels[0][-1].isnumeric():
                y_labels = ['%.3f' % float(y_label) for y_label in y_labels]
        ax[s%2][s//2].set_title(fig_text[scale], loc="right")
        ax[s%2][s//2].set_xticks([], [])
        ax[s%2][s//2].set_yticks([], [])
        ax[-1][0].set_xticks(x_indices, x_labels, rotation=45)
        ax[-1][1].set_xticks(x_indices, x_labels, rotation=45)
        ax[1][0].set_ylabel(params[yaxis])
        ax[0][0].set_ylabel(params[yaxis])
        ax[1][0].set_yticks(y_indices, y_labels)
        ax[0][0].set_yticks(y_indices, y_labels)
        ax[-1][0].set_xlabel(params[xaxis])
        ax[-1][1].set_xlabel(params[xaxis])
        ax[s%2][s//2].imshow(content, vmin=0, vmax=2, aspect="equal")
        num_buckets = len(param_vals[params[xaxis]])
        print(num_buckets)
        if scale != 1.0:
            # print(opts[scale][:,0])
            # print(opts[scale][:,1])
            ax[s%2][s//2].scatter(opts[scale][:, 0]*num_buckets - 1, opts[scale][:, 1], marker=symbol[scale], s=50, c=scale_color[scale],
                          edgecolors="black")
        else:
            for sec_scale in scales:
                if sec_scale != 0.01:
                    ax[s%2][s//2].scatter(scale_opts[sec_scale][:, 0]*num_buckets - 1, scale_opts[sec_scale][:, 1], marker=symbol[sec_scale], s=50,
                                  c=scale_color[sec_scale], edgecolors="black")

    plt.tight_layout()
    plt.savefig(f"grid_figures/{ds}_{method}_{metric}_grid_4x4.pdf", bbox_inches="tight")
    plt.show()


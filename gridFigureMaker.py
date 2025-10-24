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

    method = "em"
    metric = "sup"
    ds = "complex9"
    yaxis = 2
    xaxis = 0
    scales = [0.01, 0.1, 0.5, 1.0]

    params = []
    defaults = {}
    max_val = {}
    if method == "dbscan":
        params = ["eps", "min_samples"]
        file_name = "eps_min_samples_none"
        defaults = {"eps": "0.5", "min_samples": "5"}
        max_val = {"eps": 0.24, "min_samples": 20}
        size = (3.5, 8)
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
        size = (3.5, 3.5)
    for scale in scales:
        if not os.path.exists(f"grid_dicts/{ds}_{scale}_{method}_{metric}_{file_name}_dict.npy"):
            files = [f"grid_evals/{ds}_random_{scale}_{method}_{file_name}_100.txt"]
            file_parser(method, files, "grid_dicts", f"{ds}_{scale}_{method}_{file_name}")

    #https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    fig, ax = plt.subplots(len(scales), figsize=size, sharey=True)

    opts = {0.01: np.array([[3, 2], [3, 0], [6, 2]]),
            0.1: np.array([[3, 2], [3, 0], [4, 2], [6, 0]]),
            0.5: np.array([[3, 0], [4, 2]]),
            1.0: np.array([[]])
            }
    scale_color = {0.01: "red", 0.1: "orange", 0.5: "yellow", 1.0: "white"}
    scale_opts = {0.01: np.array([[3, 2],  [3, 0], [6, 2]]),
                  0.1: np.array([[3, 2], [3, 0], [4, 2], [6, 0]]),
                  0.5: np.array([[3, 0], [4, 2]]),
                  1.0: np.array([[5, 2], [4, 2]]),
                  }
    # for scale in scales:
    #     opt = opts[scale]
    #     for opti in range(len(opt)):
    #         print(opt[opti])


    symbol = {0.01: "X", 0.1: "P", 0.5: "*", 1.0: "."}

    #fig_text_symbol = {0.01:"╳" , 0.1: "+", 0.5: "•", 1.0: "⬟"}

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
        if len(yvals) > 10:
            y_indices = np.linspace(0, len(yvals), endpoint=False, num=10, dtype=int)
            y_labels = np.array(yvals)[y_indices]
            if not y_labels[0].isnumeric() and y_labels[0][-1].isnumeric():
                y_labels = ['%.3f' % float(y_label) for y_label in y_labels]
        else:
            y_indices = range(len(yvals))
            y_labels = yvals
            if not y_labels[0].isnumeric() and y_labels[0][-1].isnumeric():
                y_labels = ['%.3f' % float(y_label) for y_label in y_labels]
        ax[s].set_title(fig_text[scale], loc="right")
        ax[s].set_yticks(y_indices, y_labels)
        ax[s].set_xticks([], [])
        ax[-1].set_xticks(x_indices, x_labels, rotation=45)
        ax[s].imshow(content, vmin=0, vmax=2, aspect="equal")
        #print(content)
        if scale != 1.0:
            #print(opts[scale][:,0])
            #print(opts[scale][:,1])
            ax[s].scatter(opts[scale][:,0]-1, opts[scale][:,1], marker=symbol[scale], s=50, c=scale_color[scale], edgecolors="black")
        else:
            for sec_scale in scales:
                ax[s].scatter(scale_opts[sec_scale][:, 0] - 1, scale_opts[sec_scale][:, 1], marker=symbol[sec_scale], s=50, c=scale_color[sec_scale],
                              edgecolors="black")
    #https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    fig.text(0, 0.5, params[yaxis], va='center', rotation='vertical')
    plt.xlabel(params[xaxis])
    plt.tight_layout()
    plt.savefig(f"grid_figures/{ds}_{method}_{metric}_grid.pdf", bbox_inches="tight")
    plt.show()


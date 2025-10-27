import math
import os
import warnings

import numpy as np

warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)

from pprint import pprint

# gives rank per subsampling method for all methods for specific algorithm and supervision based on eval_logs
# missing results are ranked last if skip_not_found is False, else the dataset subset sizes that have a missing value are skipped
def rank_subsets(alg, supervision, skip_not_found=False):

    datasets = ["complex9", "aggregation", "densired", "densired_noise", "har", "isolet", "magic_gamma", "wine_quality", "pendigits"]
    datasets_name = {"complex9": "Complex-9", "aggregation": "Aggregation", "har": "HAR",
                     "isolet": "Isolet", "densired": "DENSIRED", "densired_noise": "DENSIRED$_{N}$",
                     "magic_gamma": "Magic-Gamma", "wine_quality": "Wine-Quality",
                     "pendigits": "Pendigits", "letter": "Letter"
                     }

    supervision_name = {"True": "supervised score (ARI + AMI)", "False": "unsupervised score (Silhouette Coefficient + DISCO)"}
    supervision_string = {"True": "supervised", "False": "unsupervised"}

    sampling_name = {"random":"Random", "lwc":"LWC", "protras": "ProTraS", "kcentroid": "k-Centroid", "dendis": "DenDis", "dides": "DiDes"}

    alg_name = {"dbscan": "DBSCAN", "kmeans": "k-Means", "spectral": "Spectral Clustering", "em": "Expectation–Maximization", "agglomerative": "Agglomerative Clustering", "dpc": "Density Peak Clustering", "hdbscan": "HDBSCAN"}

    scores = {}
    rank_per_sample = {}

    for dataset in datasets:
        not_found = False
        for i in ["0.01", "0.1", "0.25", "0.5", "0.75"]:
            prefix = f"eval_logs/{dataset}_{alg}/log_{dataset}_{alg}_{supervision_string[supervision]}_from_"
            postfix = f"_on_kcentroid_1.0_3600_sample_mult.txt"
            subscores = []
            for sampling in ["random", "lwc", "kcentroid", "protras", "dendis", "dides"]:
                key = dataset + "_" + sampling + "_" + i
                path = f"{prefix}{sampling}_{i}{postfix}"
                subscore = -np.inf
                if os.path.exists(path):
                    with open(path) as file:
                        lines = file.readlines()
                        if len(lines) > 0:
                            scoring = lines[-1].replace("\n", "")
                            scoringsplit = scoring.split(" ")
                            if scoringsplit[-4].replace('e','',1).replace('.','',1).replace('-','',2).isnumeric():
                                subscore = round((2-float(scoringsplit[-4]))*100)
                            else:
                                not_found = True
                        else:
                            not_found = True
                else:
                    not_found = True
                scores[key] = subscore
                subscores.append(subscore)
            if not not_found or not skip_not_found:
                sorted_subscores = np.sort(subscores)
                ranking = []
                for j in range(len(subscores)):
                    ranking.append(6-np.max(np.where(sorted_subscores == subscores[j])))
                #print(ranking, subscores)
                rank_per_sample[f"{dataset}_{i}"] = ranking
    #overall_ranking = np.sum(rank_per_sample.values(), axis=0)
    #print(np.sum(list(rank_per_sample.values()), axis=0)/len(list(rank_per_sample.keys())))
    return rank_per_sample

# gives ranking of subsampling strategies
if __name__ == '__main__':
    all_rankings = []
    part_rankings = []
    for alg in ["dbscan", "kmeans", "em"]:#, "spectral", "agglomerative"]:#, "hdbscan", "spectral", "dpc", "agglomerative", "em"]:
        ranks = rank_subsets(alg, "True")
        ranking = np.sum(list(ranks.values()), axis=0)/len(list(ranks.keys()))
        print(ranking)
        all_rankings.append(ranking)
        part_rankings.append(ranking)
    print("Supervised", np.sum(part_rankings, axis=0) / len(part_rankings))
    part_rankings = []
    for alg in ["dbscan", "kmeans", "em"]:#, "spectral", "agglomerative"]:#"hdbscan", "spectral", "dpc", "agglomerative", "em"]:
        ranks = rank_subsets(alg, "False")
        ranking = np.sum(list(ranks.values()), axis=0)/len(list(ranks.keys()))
        print(ranking)
        all_rankings.append(ranking)
        part_rankings.append(ranking)
    print("Unsupervised", np.sum(part_rankings, axis=0) / len(part_rankings))
    print("All", np.sum(all_rankings, axis=0) / len(all_rankings))





import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
    adjusted_mutual_info_score, davies_bouldin_score, calinski_harabasz_score
from clustpy.metrics import unsupervised_clustering_accuracy, purity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

from clustering.density_peak_clustering import DensityPeakClustering
from metrics.disco import disco_score
from datetime import datetime
import random

# runs the actual clustering and returns the clustering result
def perform_clustering(data, algorithm, config, seed):
    if algorithm == "dbscan":
        config = {"eps": 0.5, "min_samples": 5} | config
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        clustering = dbscan.fit_predict(data, None)
    elif algorithm == "kmeans":
        config = {"n_clusters": 8, "init": "k-means++"} | config
        kmeans = KMeans(n_clusters=config["n_clusters"], init=config["init"], random_state=seed)
        clustering = kmeans.fit_predict(data, None)
    elif algorithm == "spectral":
        config = {"n_clusters": 8, "affinity":'rbf', "n_neighbors": 10, "gamma": 1.0, "assign_labels": 'kmeans'} | config
        spectral = SpectralClustering(n_clusters=config["n_clusters"], gamma=config["gamma"], n_neighbors=config["n_neighbors"],
                                      assign_labels=config["assign_labels"], affinity=config["affinity"], random_state=seed)
        clustering = spectral.fit_predict(data, None)
    elif algorithm == "hdbscan":
        config = {"min_cluster_size": 5, "min_samples": 5, "cluster_selection_epsilon": 0, "alpha": 1,
                  "cluster_selection_method": "eom", "max_cluster_size": None} | config
        hdbscan = HDBSCAN(min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"],
                          cluster_selection_epsilon=config["cluster_selection_epsilon"],
                          alpha=config["alpha"], cluster_selection_method=config["cluster_selection_method"],
                          max_cluster_size=config["max_cluster_size"])
        clustering = hdbscan.fit_predict(data, None)
    elif algorithm == "agglomerative":
        config = {"n_clusters": 8, "connectivity": None, "linkage": "ward"} | config
        if config["connectivity"] == "kneighbors_graph":
            config["connectivity"] = kneighbors_graph(data, n_neighbors=config["n_neighbors"])
        agglomerative = AgglomerativeClustering(n_clusters=config["n_clusters"],connectivity=config["connectivity"],
                                                linkage=config["linkage"])
        clustering = agglomerative.fit_predict(data, None)
    elif algorithm == "dpc":
        config = {"metric": "euclidean", "use_distance_threshold": False, "distance_threshold": None,
                  "gaussian": True, "use_min_rho": False, "min_rho": None, "use_min_delta": False, "min_delta":None,
                  "hard_assign": False, "halo": True, "halo_avg": True, "halo_noise": True
                  } | config
        #print(config)
        dpc = DensityPeakClustering(metric=config["metric"],distance_threshold=config["distance_threshold"],
                                    gaussian=config["gaussian"],min_rho=config["min_rho"],min_delta=config["min_delta"],
                                    hard_assign=config["hard_assign"], halo=config["halo"],halo_avg=config["halo_avg"],
                                    halo_noise=config["halo_noise"])
        clustering = dpc.fit_predict(data, None)
    elif algorithm == "em":
        config = {"n_components": 1, "init_params": "kmeans", "covariance_type": "full"} | config
        em = GaussianMixture(n_components=config["n_components"], init_params=config["init_params"], covariance_type=config["covariance_type"],
                             random_state=seed)
        clustering = em.fit_predict(data, None)
    else:
        raise NotImplementedError
    label_max = max(np.unique(clustering)) + 1
    for i in range(len(clustering)):
        if clustering[i] == -1:
            clustering[i] = label_max
            label_max += 1

    return clustering


def eval_clustering_supervised(clustering, labels):
    acc = float(unsupervised_clustering_accuracy(labels, clustering))
    ari = float(adjusted_rand_score(labels, clustering))
    ami = float(adjusted_mutual_info_score(labels, clustering))
    nmi = float(normalized_mutual_info_score(labels, clustering))
    pur = float(purity(labels, clustering))
    return {"Accuracy": acc, "ARI": ari, "AMI": ami, "NMI": nmi, "Purity": pur}


def eval_clustering_unsupervised(clustering, data):
    try:
        sil = float(silhouette_score(data, clustering))
    except:
        sil = -1.0
    try:
        db = float(davies_bouldin_score(data, clustering))
    except:
        db = -1.0
    try:
        ch = float(calinski_harabasz_score(data, clustering))
    except:
        ch = -1.0
    dis5 = float(disco_score(data, clustering, min_points=5))
    dis10 = float(disco_score(data, clustering, min_points=10))
    return {"SilhouetteScore": sil, "DaviesBouldinScore": db, "CalinskiHarabaszScore": ch, "DISCO5": dis5,
            "DISCO10": dis10}

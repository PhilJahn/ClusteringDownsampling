import math

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition
from sklearn.neighbors import kneighbors_graph

class_num = {}
class_num["complex9"] = 9
class_num["diamond9"] = 9
class_num["letter"] = 26
class_num["EEG Eye State"] = 2

def get_configspace(method, ds, data_points):
    config_space = ConfigurationSpace()
    if ds in class_num.keys():
        num_classes = class_num[ds]
    else:
        raise NotImplementedError(ds)
    max_nn = max(((math.floor((math.log2(len(data_points))))**2) + 5), 100)
    d = len(data_points[0])

    if method == "dbscan":
        dbscan_1 = Float("eps", (0, (d**0.5)/2), default=0.5)
        dbscan_2 = Integer("min_samples", (1, 100), default=5)
        config_space.add([dbscan_1, dbscan_2])
    elif method == "kmeans":
        kmeans_1 = Integer("n_clusters", (1, 100), default=num_classes)
        kmeans_2 = Categorical("init", ["k-means++", "random"], default="k-means++")
        config_space.add([kmeans_1, kmeans_2])
    elif method == "spectral" or method == "spectral_gamma":
        spectral_1 = Integer("n_clusters", (1, 100), default=num_classes)
        spectral_2 = Float("gamma", (0, 10), default=1.0)
        spectral_3 = Categorical("assign_labels", ["kmeans", "discretize", "cluster_qr"], default="kmeans")
        config_space.add([spectral_1, spectral_2, spectral_3])
    elif method == "spectral_nn":
        spectral_1 = Integer("n_clusters", (1, 100), default=num_classes)
        spectral_2 = Integer("n_neighbors", (1, 100), default=10)
        spectral_3 = Categorical("assign_labels", ["kmeans", "discretize", "cluster_qr"], default="kmeans")
        config_space.add([spectral_1, spectral_2, spectral_3])
    elif method == "hdbscan":
        hdbscan_1 = Integer("min_cluster_size", (1, 100), default=5)
        hdbscan_2 = Integer("min_samples", (1, 100), default=5)
        hdbscan_3 = Float("cluster_selection_epsilon", (0, (d**0.5)/2), default=0)
        hdbscan_4 = Float("alpha", (0, 1), default=1)
        hdbscan_5 = Categorical("cluster_selection_method", ["eom", "leaf"], default="eom")
        config_space.add([hdbscan_1, hdbscan_2, hdbscan_3, hdbscan_4, hdbscan_5])
    elif method == "agglomerative":
        agglomerative_1 = Integer("n_clusters", (1, 100), default=num_classes)
        agglomerative_2 = Categorical("connectivity", [None, "kneighbors_graph"], default=None)
        agglomerative_3 = Integer("n_neighbors", (1, 100), default=10)
        agglomerative_4 = Categorical("linkage", ["ward", "complete", "average", "single"], default="ward")
        config_space.add([agglomerative_1, agglomerative_2, agglomerative_3, agglomerative_4])
        agglomerative_cond_1 = EqualsCondition(config_space['n_neighbors'], config_space['connectivity'], "kneighbors_graph")
        config_space.add(agglomerative_cond_1)
    else:
        raise NotImplementedError(method)
    return config_space
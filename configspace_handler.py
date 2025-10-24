import math

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical, Constant, \
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenValueError, EqualsCondition
from sklearn.neighbors import kneighbors_graph

from data_handler import load_data
from subset_handler import load_subset

class_num = {}
class_num["complex9"] = 9
class_num["complex92"] = 9
class_num["diamond9"] = 9
class_num["letter"] = 26
class_num["EEG Eye State"] = 2
class_num["sensorless"] = 11
class_num["aggregation"] = 7
class_num["aggregation2"] = 7
class_num["shuttle"] = 7
class_num["har"] = 6
class_num["magic_gamma"] = 2
class_num["isolet"] = 26
class_num["wine_quality"] = 7
class_num["pendigits"] = 10
class_num["scaling1"] = 10
class_num["scaling2"] = 10
class_num["scaling3"] = 10
class_num["scaling4"] = 10
class_num["large"] = 10
class_num["verylarge"] = 10
class_num["verylarge3"] = 20

def get_configspace(method, ds, args):
    config_space = ConfigurationSpace()
    if ds in class_num.keys():
        num_classes = class_num[ds]
    else:
        raise NotImplementedError(ds)

    data_points, labels = load_data(args.ds)
    d = len(data_points[0])

    base_1 = Constant("method", method)
    data_1 = Constant("ds", ds)
    data_2 = Constant("size", args.size)
    data_3 = Constant("sampling", args.sampling)
    data_4 = Constant("data_seed", args.data_seed)
    base_2 = Constant("supervised", args.supervised)

    config_space.add([base_1, base_2, data_1, data_2, data_3, data_4])

    if method == "dbscan":
        dbscan_1 = Float("eps", (0, (d**0.5)/2), default=0.5)
        dbscan_2 = Integer("min_samples", (1, 100), default=5)
        config_space.add([dbscan_1, dbscan_2])
    elif method == "kmeans":
        kmeans_1 = Integer("n_clusters", (1, 100), default=num_classes)
        kmeans_2 = Categorical("init", ["k-means++", "random"], default="k-means++")
        config_space.add([kmeans_1, kmeans_2])
    elif method == "spectral":
        spectral_1 = Integer("n_clusters", (1, 100), default=num_classes)
        spectral_2 = Categorical("affinity", ["rbf", "nearest_neighbors"], default="rbf")
        spectral_3 = Float("gamma", (0, 10), default=1.0)
        spectral_4 = Integer("n_neighbors", (1, 100), default=10)
        spectral_5 = Categorical("assign_labels", ["kmeans", "discretize", "cluster_qr"], default="kmeans")
        config_space.add([spectral_1, spectral_2, spectral_3, spectral_4, spectral_5])
        spectral_cond_1 = EqualsCondition(config_space['gamma'], config_space['affinity'], "rbf")
        spectral_cond_2 = EqualsCondition(config_space['n_neighbors'], config_space['affinity'], "nearest_neighbors")
        config_space.add(spectral_cond_1)
        config_space.add(spectral_cond_2)
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
    elif method == "dpc":
        dpc_1 = Constant("metric", "euclidean")
        dpc_2 = Categorical("use_distance_threshold", [True, False], default=False)
        dpc_3 = Float("distance_threshold", (0, (d**0.5)/2), default=0.5)
        dpc_4 = Categorical("gaussian", [True, False], default=True)
        dpc_5 = Categorical("use_min_rho", [True, False], default=False)
        dpc_6 = Float("min_rho", (1, 100), default=5)
        dpc_7 = Categorical("use_min_delta", [True, False], default=False)
        dpc_8 = Float("min_delta", (0, (d**0.5)/2), default=0.5)
        dpc_9 = Categorical("hard_assign", [True, False], default=False)
        dpc_10 = Categorical("halo", [True, False], default=True)
        dpc_11 = Categorical("halo_avg", [True, False], default=True)
        config_space.add([dpc_1,dpc_2,dpc_3,dpc_4,dpc_5,dpc_6,dpc_7,dpc_8,dpc_9,dpc_10,dpc_11])
        dpc_cond_1 = EqualsCondition(config_space['distance_threshold'], config_space['use_distance_threshold'], True)
        dpc_cond_2 = EqualsCondition(config_space['min_rho'], config_space['use_min_rho'], True)
        dpc_cond_3 = EqualsCondition(config_space['min_delta'], config_space['use_min_delta'], True)
        dpc_cond_4 = EqualsCondition(config_space['halo_avg'], config_space['halo'], True)

        config_space.add(dpc_cond_1)
        config_space.add(dpc_cond_2)
        config_space.add(dpc_cond_3)
        config_space.add(dpc_cond_4)
    elif method == "em":
        em_1 = Integer("n_components", (1, 100), default=num_classes)
        em_2 = Categorical("init_params", ["kmeans", "k-means++", "random", "random_from_data"], default="kmeans")
        em_3 = Categorical("covariance_type", ["full", "tied", "diag", "spherical"], default="full")
        config_space.add([em_1, em_2, em_3])
    else:
        raise NotImplementedError(method)
    return config_space
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
    adjusted_mutual_info_score, davies_bouldin_score, calinski_harabasz_score
from clustpy.metrics import unsupervised_clustering_accuracy, purity
from metrics.disco import disco_score
from datetime import datetime
import random

def perform_clustering(data, algorithm, config, seed):
    autoencoder = None

    #----Traditional Methods----
    if algorithm == "dbscan":
        config = {"eps": 0.5, "min_samples": 5} | config
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        clustering = dbscan.fit_predict(data, None)
    elif algorithm == "kmeans":
        config = {"init": "euclidean"} | config
        kmeans = KMeans(n_clusters=config["n_clusters"], init=config["init"], random_state=seed)
        clustering = kmeans.fit_predict(data, None)
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
    return {"SilhouetteScore": sil, "DaviesBouldinScore": db, "CalinskiHarabaszScore": ch, "DISCO5": dis5, "DISCO10": dis10}

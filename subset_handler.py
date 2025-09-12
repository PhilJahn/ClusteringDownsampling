import argparse
import os

import numpy as np
from numpy.random import PCG64
from sklearn.cluster import KMeans, Birch
from sklearn.metrics.pairwise import distance_metrics

from data_handler import load_data


def random_subsampling(dataset, ratio, seed=0):
    data, labels = load_data(dataset)
    num = round(len(labels) * ratio)
    generator = np.random.Generator(PCG64(seed))
    rand_subset_indices = np.sort(generator.choice(len(labels), size=num, replace=False))
    X_rand_subset, y_rand_subset = data[rand_subset_indices], labels[rand_subset_indices]

    if not os.path.exists("./data/rand_subset"):
        os.mkdir("./data/rand_subset")
    np.save(f"./data/rand_subset/data_{dataset}_random_{ratio}_{seed}.npy", X_rand_subset)
    np.save(f"./data/rand_subset/labels_{dataset}_random_{ratio}_{seed}.npy", y_rand_subset)


def kmeans_subsampling(dataset, ratio, seed=0):
    data, labels = load_data(dataset)
    num = round(len(labels) * ratio)
    kmeans = KMeans(n_clusters=num, random_state=seed)

    clustering = kmeans.fit_predict(data)
    #print(clustering)

    generator = np.random.Generator(PCG64(seed))

    #plt.figure(figsize=(10, 10))
    #plt.scatter(data[:, 0], data[:, 1], c=clustering)
    #plt.show()

    subset_indices = []
    for cluster in np.unique(clustering):
        candidates = np.where(clustering == cluster)[0]
        chosen = generator.choice(candidates)
        #print(cluster, candidates,chosen)
        subset_indices.append(chosen)
    X_kmeans_subset = data[subset_indices]
    y_kmeans_subset = labels[subset_indices]

    if not os.path.exists("./data/kmeans_subset"):
        os.mkdir("./data/kmeans_subset")
    np.save(f"./data/kmeans_subset/data_{dataset}_kmeans_{ratio}_{seed}.npy", X_kmeans_subset)
    np.save(f"./data/kmeans_subset/labels_{dataset}_kmeans_{ratio}_{seed}.npy", y_kmeans_subset)

def kcentroid_subsampling(dataset, ratio, seed=0):
    data, labels = load_data(dataset)
    num = round(len(labels) * ratio)
    kmeans = KMeans(n_clusters=num, random_state=seed)

    clustering = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_

    X_kcentroid_subset = centers
    y_kcentroid_subset = []

    subset_indices = []
    for i in range(len(centers)):
        clustered = np.where(clustering == i)[0]
        cluster_labels = [0]*len(np.unique(labels))
        _, mapping = np.unique(labels, return_inverse=True)
        for c_id in clustered:
            cluster_labels[mapping[c_id]] += 1
        cluster_label = np.argmax(cluster_labels)
        y_kcentroid_subset.append(cluster_label)

    X_kcentroid_subset = np.array(X_kcentroid_subset)
    y_kcentroid_subset = np.array(y_kcentroid_subset)

    if not os.path.exists("./data/kcentroid_subset"):
        os.mkdir("./data/kcentroid_subset")
    np.save(f"./data/kcentroid_subset/data_{dataset}_kcentroid_{ratio}_{seed}.npy", X_kcentroid_subset)
    np.save(f"./data/kcentroid_subset/labels_{dataset}_kcentroid_{ratio}_{seed}.npy", y_kcentroid_subset)

# from Scalable k-Means Clustering via Lightweight Coresets
def lwc_subsampling(dataset, ratio, seed=0, distance_function="euclidean"):
    data, labels = load_data(dataset)
    mu = np.mean(data, axis=0).reshape(1, -1)

    distance_metric = distance_metrics()[distance_function]
    dists = distance_metric(mu,data)[0]
    #print(dists)
    probs = []
    divisor = np.sum(dists)
    #print(divisor)
    for i in range(len(data)):
        prob = 0.5/len(data) + 0.5*dists[i]/divisor
        #print(prob)
        probs.append(prob)
    #print(probs)
    num = round(len(labels) * ratio)
    generator = np.random.Generator(PCG64(seed))
    lwc_subset_indices = np.sort(generator.choice(len(data),size=num, replace=False, p=probs))
    X_lwc_subset, y_lwc_subset = data[lwc_subset_indices], labels[lwc_subset_indices]

    if not os.path.exists("./data/lwc_subset"):
        os.mkdir("./data/lwc_subset")
    np.save(f"./data/lwc_subset/data_{dataset}_lwc_{distance_function}_{ratio}_{seed}.npy", X_lwc_subset)
    np.save(f"./data/lwc_subset/labels_{dataset}_lwc_{distance_function}_{ratio}_{seed}.npy", y_lwc_subset)

# from ProTraS: A probabilistic traversing sampling algorithm, altered to use ratio rather than epsilon for consistency with other methods
def protras_subsampling_old(dataset, ratio, seed=0, distance_function="euclidean"):
    data, labels = load_data(dataset)
    distance_metric = distance_metrics()[distance_function]
    num = round(len(labels) * ratio)

#data = T, pattern = T(y), t_n_s = T \ S
    pattern = {}
    t_n_s = list(range(len(data)))


    virtual_start = np.min(data,axis=0).reshape(1, -1)
    #print(virtual_start)
    closest_i = -1
    closest_distance = np.inf

    for i in range(len(data)):
        dist = distance_metric(virtual_start,data[i].reshape(1, -1))[0][0]
        if dist < closest_distance:
            closest_distance = dist
            closest_i = i
    s = [closest_i]
    pattern[closest_i] = [closest_i]
    t_n_s.remove(closest_i)

    dist_matrix = distance_metric(data, data) # no recalculation of distances in every loop

    while len(s) < num:
        for l in t_n_s:
            min_dist_l = np.inf
            min_k = -1
            for k in s:
                dist_l = dist_matrix[l,k]
                if dist_l < min_dist_l:
                    min_dist_l = dist_l
                    min_k = k
            pattern[min_k].append(l)
        maxwd = 0
        ystar = -1
        xmax = -1
        for k in s:
            max_dist_k = 0
            max_m = -1
            for m in pattern[k]:
                dist_m = dist_matrix[m,k]
                if dist_m > max_dist_k:
                    max_dist_k = dist_m
                    max_m = m
                pk = len(pattern[k]) * max_dist_k
                if maxwd < pk:
                    maxwd = pk
                    ystar = k
                    xmax = max_m
        s.append(xmax)
        #print(len(s), s, pattern)
        for i in s: #full reset of pattern, otherwise always stuck on first candidate
            pattern[i] = [i]
        t_n_s.remove(xmax)

    X_protras_subset, y_protras_subset = data[s], labels[s]
    if not os.path.exists("./data/protras_subset"):
        os.mkdir("./data/protras_subset")
    np.save(f"./data/protras_subset/data_{dataset}_protras_{distance_function}_{ratio}_{seed}.npy", X_protras_subset)
    np.save(f"./data/protras_subset/labels_{dataset}_protras_{distance_function}_{ratio}_{seed}.npy", y_protras_subset)

def protras_subsampling(dataset, ratio, seed=0, distance_function="euclidean"):
    data, labels = load_data(dataset)
    distance_metric = distance_metrics()[distance_function]
    num = round(len(labels) * ratio)

#data = T, pattern = T(y), t_n_s = T \ S
    pattern = {}
    t_n_s = list(range(len(data)))


    virtual_start = np.min(data,axis=0).reshape(1, -1)
    #print(virtual_start)
    closest_i = -1
    closest_distance = np.inf

    for i in range(len(data)):
        dist = distance_metric(virtual_start,data[i].reshape(1, -1))[0][0]
        if dist < closest_distance:
            closest_distance = dist
            closest_i = i
    s = [closest_i]
    pattern[closest_i] = []
    rev_pattern = {}
    t_n_s.remove(closest_i)

    cur_k = closest_i
    for l in range(len(data)):
        rev_pattern[l] = cur_k
        pattern[cur_k].append(l)
    dist_matrix = {}
    dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[0] # no recalculation of distances in every loop
    #print(dist_matrix[cur_k])

    while len(s) < num:
        for l in t_n_s: # only update if newest it closer -> no n^2 loop here
            if dist_matrix[cur_k][l] <= dist_matrix[rev_pattern[l]][l]: # no impact over <, but addresses first loop
                pattern[rev_pattern[l]].remove(l)
                pattern[cur_k].append(l)
                rev_pattern[l] = cur_k
        maxwd = 0
        xmax = -1
        ymax = -1
        for k in s:
            max_dist_k = 0
            max_m = -1
            for m in list(set(pattern[k]).intersection(set(t_n_s))):
                dist_m = dist_matrix[k][m]
                if dist_m > max_dist_k:
                    max_dist_k = dist_m
                    max_m = m
                pk = len(pattern[k]) * max_dist_k
                if maxwd < pk:
                    maxwd = pk
                    ymax = k
                    xmax = max_m
        cur_k = xmax
        s.append(cur_k)
        #print(len(s), s, pattern)
        pattern[rev_pattern[cur_k]].remove(cur_k)
        pattern[cur_k] = [cur_k]
        rev_pattern[cur_k] = cur_k
        t_n_s.remove(cur_k)
        dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[0]


    X_protras_subset, y_protras_subset = data[s], labels[s]
    if not os.path.exists("./data/protras_subset"):
        os.mkdir("./data/protras_subset")
    np.save(f"./data/protras_subset/data_{dataset}_protras_{distance_function}_{ratio}_{seed}.npy", X_protras_subset)
    np.save(f"./data/protras_subset/labels_{dataset}_protras_{distance_function}_{ratio}_{seed}.npy", y_protras_subset)

def dendis_subsampling(dataset, ratio, seed=0, distance_function="euclidean"):
    data, labels = load_data(dataset)
    distance_metric = distance_metrics()[distance_function]
    num = round(len(labels) * ratio)

#data = T, pattern = T(y), t_n_s = T \ S
    pattern = {}
    t_n_s = list(range(len(data)))


    virtual_start = np.min(data,axis=0).reshape(1, -1)
    #print(virtual_start)
    closest_i = -1
    closest_distance = np.inf

    for i in range(len(data)):
        dist = distance_metric(virtual_start,data[i].reshape(1, -1))[0][0]
        if dist < closest_distance:
            closest_distance = dist
            closest_i = i
    s = [closest_i]
    pattern[closest_i] = []
    rev_pattern = {}
    t_n_s.remove(closest_i)

    cur_k = closest_i
    for l in range(len(data)):
        rev_pattern[l] = cur_k
        pattern[cur_k].append(l)
    dist_matrix = {}
    dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[0] # no recalculation of distances in every loop
    #print(dist_matrix[cur_k])

    while len(s) < num:
        for l in t_n_s: # only update if newest it closer -> no n^2 loop here
            if dist_matrix[cur_k][l] <= dist_matrix[rev_pattern[l]][l]: # no impact over <, but addresses first loop
                pattern[rev_pattern[l]].remove(l)
                pattern[cur_k].append(l)
                rev_pattern[l] = cur_k
        maxsize = -1
        maxk = -1
        for k in s:
            if len(pattern[k]) > maxsize:
                maxsize = len(pattern[k])
                maxk = k
        max_dist_k = -1
        max_m = -1
        for m in list(set(pattern[maxk]).intersection(set(t_n_s))):
            dist_m = dist_matrix[maxk][m]
            if dist_m > max_dist_k:
                max_dist_k = dist_m
                max_m = m
        cur_k = max_m
        s.append(cur_k)
        #print(len(s), s, pattern)
        pattern[rev_pattern[cur_k]].remove(cur_k)
        pattern[cur_k] = [cur_k]
        rev_pattern[cur_k] = cur_k
        t_n_s.remove(cur_k)
        dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[0]


    X_dendis_subset, y_dendis_subset = data[s], labels[s]
    if not os.path.exists("./data/dendis_subset"):
        os.mkdir("./data/dendis_subset")
    np.save(f"./data/dendis_subset/data_{dataset}_dendis_{distance_function}_{ratio}_{seed}.npy", X_dendis_subset)
    np.save(f"./data/dendis_subset/labels_{dataset}_dendis_{distance_function}_{ratio}_{seed}.npy", y_dendis_subset)

def dides_subsampling(dataset, ratio, seed=0, distance_function="euclidean"):
    data, labels = load_data(dataset)
    distance_metric = distance_metrics()[distance_function]
    num = round(len(labels) * ratio)

#data = T, pattern = T(y), t_n_s = T \ S
    pattern = {}
    t_n_s = list(range(len(data)))


    virtual_start = np.min(data,axis=0).reshape(1, -1)
    #print(virtual_start)
    closest_i = -1
    closest_distance = np.inf

    for i in range(len(data)):
        dist = distance_metric(virtual_start,data[i].reshape(1, -1))[0][0]
        if dist < closest_distance:
            closest_distance = dist
            closest_i = i
    s = [closest_i]
    pattern[closest_i] = []
    rev_pattern = {}
    t_n_s.remove(closest_i)

    cur_k = closest_i
    for l in range(len(data)):
        rev_pattern[l] = cur_k
        pattern[cur_k].append(l)
    dist_matrix = {}
    dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[0] # no recalculation of distances in every loop
    #print(dist_matrix[cur_k])

    while len(s) < num:
        for l in t_n_s: # only update if newest it closer -> no n^2 loop here
            if dist_matrix[cur_k][l] <= dist_matrix[rev_pattern[l]][l]: # no impact over <, but addresses first loop
                pattern[rev_pattern[l]].remove(l)
                pattern[cur_k].append(l)
                rev_pattern[l] = cur_k
        maxwd = 0
        xmax = -1
        ymax = -1
        for k in s:
            max_dist_k = 0
            max_m = -1
            for m in list(set(pattern[k]).intersection(set(t_n_s))):
                dist_m = dist_matrix[k][m]
                if dist_m > max_dist_k:
                    max_dist_k = dist_m
                    max_m = m
                pk = max_dist_k
                if maxwd < pk:
                    maxwd = pk
                    ymax = k
                    xmax = max_m
        cur_k = xmax
        s.append(cur_k)
        #print(len(s), s, pattern)
        pattern[rev_pattern[cur_k]].remove(cur_k)
        pattern[cur_k] = [cur_k]
        rev_pattern[cur_k] = cur_k
        t_n_s.remove(cur_k)
        dist_matrix[cur_k] = distance_metric(data[cur_k].reshape(1, -1), data)[0]


    X_dides_subset, y_dides_subset = data[s], labels[s]
    if not os.path.exists("./data/dides_subset"):
        os.mkdir("./data/dides_subset")
    np.save(f"./data/dides_subset/data_{dataset}_dides_{distance_function}_{ratio}_{seed}.npy", X_dides_subset)
    np.save(f"./data/dides_subset/labels_{dataset}_dides_{distance_function}_{ratio}_{seed}.npy", y_dides_subset)

def birch_subsampling(dataset, ratio, seed):
    data, labels = load_data(dataset)
    num = round(len(labels) * ratio)
    centers = []
    threshold = 1
    while len(centers) < num:
        threshold = 0.5 * threshold
        birch = Birch(n_clusters=num, threshold=threshold)
        clustering = birch.fit_predict(data)
        # print(clustering)
        centers = birch.subcluster_centers_

    generator = np.random.Generator(PCG64(seed))
    subset_indices = []
    for cluster in np.unique(clustering):
        candidates = np.where(clustering == cluster)[0]
        chosen = generator.choice(candidates)
        #print(cluster, candidates,chosen)
        subset_indices.append(chosen)
    X_birch_subset = data[subset_indices]
    y_birch_subset = labels[subset_indices]

    if not os.path.exists("./data/birch_subset"):
        os.mkdir("./data/birch_subset")
    np.save(f"./data/birch_subset/data_{dataset}_birch_{ratio}_{seed}.npy", X_birch_subset)
    np.save(f"./data/birch_subset/labels_{dataset}_birch_{ratio}_{seed}.npy", y_birch_subset)



def load_random_subset(dataset, ratio, seed=0):
    if not os.path.exists(f"./data/rand_subset/data_{dataset}_random_{ratio}_{seed}.npy"):
        random_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/rand_subset/data_{dataset}_random_{ratio}_{seed}.npy")
    labels = np.load(f"./data/rand_subset/labels_{dataset}_random_{ratio}_{seed}.npy")

    return data, labels

def load_kmeans_subset(dataset, ratio, seed):
    if not os.path.exists(f"./data/kmeans_subset/data_{dataset}_kmeans_{ratio}_{seed}.npy"):
        kmeans_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/kmeans_subset/data_{dataset}_kmeans_{ratio}_{seed}.npy")
    labels = np.load(f"./data/kmeans_subset/labels_{dataset}_kmeans_{ratio}_{seed}.npy")

    return data, labels

def load_kcentroid_subset(dataset, ratio, seed):
    if not os.path.exists(f"./data/kcentroid_subset/data_{dataset}_kcentroid_{ratio}_{seed}.npy"):
        kcentroid_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/kcentroid_subset/data_{dataset}_kcentroid_{ratio}_{seed}.npy")
    labels = np.load(f"./data/kcentroid_subset/labels_{dataset}_kcentroid_{ratio}_{seed}.npy")

    return data, labels

def load_lwc_subset(dataset, ratio, seed, distance_function="euclidean"):
    if not os.path.exists(f"./data/lwc_subset/data_{dataset}_lwc_{distance_function}_{ratio}_{seed}.npy"):
        lwc_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/lwc_subset/data_{dataset}_lwc_{distance_function}_{ratio}_{seed}.npy")
    labels = np.load(f"./data/lwc_subset/labels_{dataset}_lwc_{distance_function}_{ratio}_{seed}.npy")

    return data, labels

def load_protras_subset(dataset, ratio, seed, distance_function="euclidean"):
    if not os.path.exists(f"./data/protras_subset/data_{dataset}_protras_{distance_function}_{ratio}_{seed}.npy"):
        protras_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/protras_subset/data_{dataset}_protras_{distance_function}_{ratio}_{seed}.npy")
    labels = np.load(f"./data/protras_subset/labels_{dataset}_protras_{distance_function}_{ratio}_{seed}.npy")

    return data, labels


def load_dendis_subset(dataset, ratio, seed, distance_function="euclidean"):
    if not os.path.exists(f"./data/dendis_subset/data_{dataset}_dendis_{distance_function}_{ratio}_{seed}.npy"):
        dendis_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/dendis_subset/data_{dataset}_dendis_{distance_function}_{ratio}_{seed}.npy")
    labels = np.load(f"./data/dendis_subset/labels_{dataset}_dendis_{distance_function}_{ratio}_{seed}.npy")

    return data, labels


def load_dides_subset(dataset, ratio, seed, distance_function="euclidean"):
    if not os.path.exists(f"./data/dides_subset/data_{dataset}_dides_{distance_function}_{ratio}_{seed}.npy"):
        dides_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/dides_subset/data_{dataset}_dides_{distance_function}_{ratio}_{seed}.npy")
    labels = np.load(f"./data/dides_subset/labels_{dataset}_dides_{distance_function}_{ratio}_{seed}.npy")

    return data, labels

def load_birch_subset(dataset, ratio, seed):
    if not os.path.exists(f"./data/birch_subset/data_{dataset}_birch_{ratio}_{seed}.npy"):
        birch_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/birch_subset/data_{dataset}_birch_{ratio}_{seed}.npy")
    labels = np.load(f"./data/birch_subset/labels_{dataset}_birch_{ratio}_{seed}.npy")

    return data, labels

def load_subset(dataset, ratio, subset_type, seed=0):
    if subset_type == "random":
        if not os.path.exists("./data/rand_subset"):
            print("Subset not found")
            os.mkdir("./data/rand_subset")
        return load_random_subset(dataset, ratio, seed)
    elif subset_type == "kmeans":
        if not os.path.exists("./data/kmeans_subset"):
            print("Subset not found")
            os.mkdir("./data/kmeans_subset")
        return load_kmeans_subset(dataset, ratio, seed)
    elif subset_type == "kcentroid":
        if not os.path.exists("./data/kcentroid_subset"):
            print("Subset not found")
            os.mkdir("./data/kcentroid_subset")
        return load_kcentroid_subset(dataset, ratio, seed)
    elif subset_type == "lwc":
        if not os.path.exists("./data/lwc_subset"):
            print("Subset not found")
            os.mkdir("./data/lwc_subset")
        return load_lwc_subset(dataset, ratio, seed, "euclidean")
    elif subset_type == "protras":
        if not os.path.exists("./data/protras_subset"):
            print("Subset not found")
            os.mkdir("./data/protras_subset")
        return load_protras_subset(dataset, ratio, seed, "euclidean")
    elif subset_type == "dendis":
        if not os.path.exists("./data/dendis_subset"):
            print("Subset not found")
            os.mkdir("./data/dendis_subset")
        return load_dendis_subset(dataset, ratio, seed, "euclidean")
    elif subset_type == "dides":
        if not os.path.exists("./data/dides_subset"):
            print("Subset not found")
            os.mkdir("./data/dides_subset")
        return load_dides_subset(dataset, ratio, seed, "euclidean")
    elif subset_type == "birch":
        if not os.path.exists("./data/birch_subset"):
            print("Subset not found")
            os.mkdir("./data/birch_subset")
        return load_birch_subset(dataset, ratio, seed)
    else:
        raise NotImplementedError

import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     dataset_name = "shuttle"
#     data, labels = load_data(dataset_name)
#     print(len(labels))
#     data, labels = load_dendis_subset(dataset_name, 0.1, 0)
#     print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_full.png", bbox_inches='tight')
    # data, labels = load_random_subset(dataset_name, 0.5, 0)
    # print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_random_subset.png", bbox_inches='tight')
    # data, labels = load_lwc_subset(dataset_name, 0.5, 0)
    # print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_lwc_subset.png", bbox_inches='tight')
    # data, labels = load_protras_subset(dataset_name, 0.5, 0)
    # print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_protras_subset.png", bbox_inches='tight')
    # data, labels = load_dendis_subset(dataset_name, 0.5, 0)
    # print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_dendis_subset.png", bbox_inches='tight')
    # data, labels = load_dides_subset(dataset_name, 0.5, 0)
    # print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_dides_subset.png", bbox_inches='tight')
    # data, labels = load_kmeans_subset(dataset_name, 0.5, 0)
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_kmeans_subset.png", bbox_inches='tight')
    # data, labels = load_kcentroid_subset(dataset_name, 0.5, 0)
    # print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_kcentroid_subset.png", bbox_inches='tight')
    # data, labels = load_birch_subset(dataset_name, 0.5, 0)
    # print(len(labels))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.savefig(f"./data_figures/{dataset_name}_birch_subset.png", bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="complex9", type=str, help='Dataset')
    parser.add_argument('--size', default=1.0, type=float, help='Size of Dataset')
    parser.add_argument('--sampling', default="kmeans", type=str, help='Downsampling Strategy')
    parser.add_argument('--data_seed', default=0, type=int, help='Seed for dataset')
    args = parser.parse_args()
    load_subset(args.ds, args.size, args.sampling, args.data_seed)
    print("done")
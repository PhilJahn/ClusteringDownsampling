import os

import numpy as np
from densired import datagen
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from clustering_handler import eval_clustering_supervised
from data_handler import load_data


def make_densired_ds():
    if not os.path.exists(f"./data"):
        os.makedirs(f"./data", exist_ok=True)
    if not os.path.exists(f"./data/synth"):
        os.makedirs(f"./data/synth", exist_ok=True)
    densired_gen = datagen.densityDataGen(dim=50, ratio_noise = 0.1, max_retry=5, dens_factors=[1,1,0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1], square=True,
                   clunum= 10, seed = 6, core_num= 200, momentum=[0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6, 0.45, 0.7],
                   branch=[0,0.05, 0.1, 0, 0, 0.1, 0.02, 0, 0, 0.25],
                   con_min_dist=0.8, verbose=True, safety=True, domain_size = 20)
    data = densired_gen.generate_data(20000)
    print(data.shape)
    with open("./data/synth/densired_noise.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)
    with open("./data/synth/densired_no_noise.txt", 'w') as f:
        for x in data:
            if x[-1] >= 0:
                strx = ""
                for xi in x:
                    strx += str(xi) + ","
                strx = strx[:-1] + "\n"
                f.write(strx)

def make_scaling_ds():
    if not os.path.exists(f"./data"):
        os.makedirs(f"./data", exist_ok=True)
    if not os.path.exists(f"./data/synth"):
        os.makedirs(f"./data/synth", exist_ok=True)
    densired_gen = datagen.densityDataGen(dim=5, ratio_noise = 0, max_retry=5, square=True,
                   clunum= 10, seed = 0, core_num= 50, verbose=True, safety=True, domain_size = 20)
    data = densired_gen.generate_data(500, seed=0)
    print(data.shape)
    with open("./data/synth/scaling_test.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)
    data2 = densired_gen.generate_data(500, seed=1)
    with open("./data/synth/scaling_test_comb.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)
        for x in data2:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)
    data = densired_gen.generate_data(1000, seed=0)
    with open("./data/synth/scaling_test_dist.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)

def make_large_ds():
    if not os.path.exists(f"./data"):
        os.makedirs(f"./data", exist_ok=True)
    if not os.path.exists(f"./data/synth"):
        os.makedirs(f"./data/synth", exist_ok=True)
    densired_gen = datagen.densityDataGen(dim=2, ratio_noise = 0, max_retry=5, square=True, momentum=None, branch="Rand", star="Rand",
                                          dens_factors=True, clunum= 10, seed = 0, core_num= 5000, verbose=True, safety=True, domain_size = 20)
    data = densired_gen.generate_data(200000, seed=0)
    print(data.shape)
    with open("./data/synth/largedens.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)

def make_verylarge_ds():
    if not os.path.exists(f"./data"):
        os.makedirs(f"./data", exist_ok=True)
    if not os.path.exists(f"./data/synth"):
        os.makedirs(f"./data/synth", exist_ok=True)
    densired_gen = datagen.densityDataGen(dim=2, ratio_noise = 0, max_retry=5, square=True, momentum=None, branch="Rand", star="Rand",
                                          dens_factors=True, clunum= 10, seed = 2, core_num= 15000, verbose=True, safety=True, domain_size = 20, min_dist = 1.5)
    data = densired_gen.generate_data(1000000, seed=0)
    print(data.shape)
    with open("./data/synth/verylargedens.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)

def make_verylargethree_ds():
    if not os.path.exists(f"./data"):
        os.makedirs(f"./data", exist_ok=True)
    if not os.path.exists(f"./data/synth"):
        os.makedirs(f"./data/synth", exist_ok=True)
    densired_gen = datagen.densityDataGen(dim=3, ratio_noise = 0, max_retry=5, square=True, momentum=None, branch="Rand", star="Rand",
                                          dens_factors=True, clunum= 20, seed = 2, core_num= 10000, verbose=True, safety=True, domain_size = 20, min_dist = 1.5)
    data = densired_gen.generate_data(800000, seed=0)
    print(data.shape)
    with open("./data/synth/verylargethreedens.txt", 'w') as f:
        for x in data:
            strx = ""
            for xi in x:
                strx += str(xi) + ","
            strx = strx[:-1] + "\n"
            f.write(strx)

if __name__ == "__main__":
    #make_densired_ds()
    #make_scaling_ds()
    #make_large_ds()
    #make_verylarge_ds()
    make_verylargethree_ds()
    ds = "scaling1"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)

    ds = "scaling2"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)

    ds = "scaling3"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)

    ds = "scaling4"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)

    ds = "large"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)

    ds = "verylarge"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)

    ds = "verylarge3"
    X, y = load_data(ds)
    print(ds, len(y), min(y), len(np.unique(y)))
    print(X.shape)

    clus = KMeans(n_clusters=20, random_state=0).fit_predict(X)
    print(eval_clustering_supervised(clus, y))
    clus = KMeans(n_clusters=22, random_state=0).fit_predict(X)
    print(eval_clustering_supervised(clus, y))
    clus2 = KMeans(n_clusters=25, random_state=0).fit_predict(X)
    print(eval_clustering_supervised(clus2, y))
    clus3 = KMeans(n_clusters=18, random_state=0).fit_predict(X)
    print(eval_clustering_supervised(clus3, y))
    clus3 = KMeans(n_clusters=15, random_state=0).fit_predict(X)
    print(eval_clustering_supervised(clus3, y))
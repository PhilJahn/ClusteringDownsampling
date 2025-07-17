import os

import numpy as np
from numpy.random import PCG64

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


def load_random_subset(dataset, ratio, seed=0):
    if not os.path.exists(f"./data/rand_subset/data_{dataset}_random_{ratio}_{seed}.npy"):
        random_subsampling(dataset, ratio, seed)
    data = np.load(f"./data/rand_subset/data_{dataset}_random_{ratio}_{seed}.npy")
    labels = np.load(f"./data/rand_subset/labels_{dataset}_random_{ratio}_{seed}.npy")

    return data, labels

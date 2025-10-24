import argparse
import os

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from clustpy.metrics import unsupervised_clustering_accuracy

from data_handler import load_data
from metrics.disco import disco_score
from similarity.BoP import BoP
from similarity.imbalance_degree import imbalance_degree
from subset_handler import load_subset
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="verylarge3", type=str, help='Dataset')
    args = parser.parse_args()
    print(args)


    data_points, labels = load_data(args.ds)
    if len(data_points) < 60000:
        try:
            sil = float(silhouette_score(data_points, labels))
        except:
            sil = -1.0
        dis5 = float(disco_score(data_points, labels, min_points=5))
        unsup_score = (sil + dis5)*100
    else:
        sil = "?"
        dis5 = "?"
        unsup_score = "?"

    counts, imb = imbalance_degree(labels)
    if not os.path.exists("data_eval"):
        os.makedirs("data_eval", exist_ok=True)


    log_name = f'data_eval/{args.ds}.txt'
    log_file = open(log_name, 'w', buffering=1)

    results = f"n:{len(data_points)}\n"
    results += f"d:{len(data_points[0])}\n"
    results += f"c:{len(np.unique(labels))}\n"
    results += f"unsup_score:{unsup_score}\n"
    results += f"sil:{sil}\n"
    results += f"dis5:{dis5}\n"
    results += f"imb:{imb}\n"


    log_file.write(results)
    log_file.close()

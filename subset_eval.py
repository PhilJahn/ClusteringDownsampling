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

# evaluates subset based on settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="scaling4", type=str, help='Dataset')
    parser.add_argument('--size', default=1.0, type=float, help='Size of Dataset')
    parser.add_argument('--sampling', default="random", type=str, help='Downsampling Strategy')
    args = parser.parse_args()
    print(args)


    if args.sampling in ["random", "lwc", "kcentroid"]:
        seeds = [0, 1, 2]
    else:
        seeds = [0]

    mmd_rbf_vals = []
    bop_jss = []
    nn_dists = []
    nn_accs = []
    unsup_scores = []
    dis5s = []
    sils = []
    imbs = []


    for data_seed in seeds:
        all_data_points, all_labels = load_data(args.ds)
        sub_data_points, sub_labels = load_subset(args.ds, args.size, args.sampling, data_seed)

        bop_all = BoP(all_data_points, min(1000, len(all_data_points)),
                      f"bop/{args.ds}/{args.sampling}/{args.size}/{data_seed}")
        bop_scores = bop_all.evaluate(sub_data_points)
        bop_js = bop_scores['JS']
        bop_jss.append(bop_js)

        #mmd_loss = MMD_loss()
        #mmd_rbf_val = mmd_loss.forward(all_data_points, sub_data_points)
        #mmd_rbf_vals.append(mmd_rbf_val)

        subset_nearest_neighbor = NearestNeighbors(n_neighbors=1).fit(sub_data_points)
        dists, nn = subset_nearest_neighbor.kneighbors(all_data_points)
        nn_dist = np.mean(dists)
        nn_dists.append(nn_dist)

        labels_nn = sub_labels[nn]
        nn_acc = unsupervised_clustering_accuracy(labels_nn, all_labels)
        nn_accs.append(nn_acc)
        try:
            sil = float(silhouette_score(sub_data_points, sub_labels))
        except:
            sil = -1.0
        sils.append(sil)
        dis5 = float(disco_score(sub_data_points, sub_labels, min_points=5))
        dis5s.append(dis5)
        unsup_score = (sil + dis5)*100
        unsup_scores.append(unsup_score)

        counts, imb = imbalance_degree(sub_labels)
        imbs.append(imb)

    if not os.path.exists("subset_eval"):
        os.makedirs("subset_eval", exist_ok=True)

    subset_log_name = f'subset_eval/{args.ds}_{args.sampling}_{args.size}.txt'
    subset_log_file = open(subset_log_name, 'w', buffering=1)

    #results = f"mmd_rbf_val:{np.mean(mmd_rbf_vals)};{np.std(mmd_rbf_vals)}\n"
    results = f"bop_js:{np.mean(bop_jss)};{np.std(bop_jss)}\n"
    results += f"nn_dist:{np.mean(nn_dists)};{np.std(nn_dists)}\n"
    results += f"nn_acc:{np.mean(nn_accs)};{np.std(nn_accs)}\n"
    results += f"unsup_score:{np.mean(unsup_scores)};{np.std(unsup_scores)}\n"
    results += f"sil:{np.mean(sils)};{np.std(sils)}\n"
    results += f"dis5:{np.mean(dis5s)};{np.std(dis5s)}\n"
    results += f"imb:{np.mean(imbs)};{np.std(imbs)}\n"


    subset_log_file.write(results)
    subset_log_file.close()

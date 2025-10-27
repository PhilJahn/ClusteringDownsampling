import math
import os
import warnings

import numpy as np
from scipy import stats
from pprint import pprint

warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)



def make_sub_table():
    dss = ["aggregation", "complex9", "densired", "densired_noise", "wine_quality", "isolet", "har", "pendigits",
           "magic_gamma", "letter"]
    dss = ["complex9", "densired", "isolet", "pendigits"]
    samplings = ["random", "lwc", "kcentroid", "protras", "dendis", "dides"]
    datasets_name = {"complex9": "Complex-9", "aggregation": "Aggregation", "har": "HAR",
                     "isolet": "Isolet", "densired": "DENSIRED", "densired_noise": "DENSIRED$_{N}$",
                     "magic_gamma": "Magic-Gamma", "wine_quality": "Wine-Quality",
                     "pendigits": "Pendigits", "letter": "Letter"
                     }
    sampling_name = {"random":"Random", "lwc":"LWC", "protras": "ProTraS", "kcentroid": "k-Centroid", "dendis": "DenDis", "dides": "DiDes"}

    sizes = [0.01, 0.1, 0.25, 0.5, 0.75]

    class_num = {"complex9": 9, "aggregation": 7, "har": 6,
                     "isolet": 26, "densired": 10, "densired_noise": 2010,
                     "magic_gamma": 2, "wine_quality": 7,
                     "pendigits": 10, "letter": 26}

    base_id = {"complex9": 5.49, "aggregation": 3.52, "har": 1.14,
                     "isolet": 1.0, "densired": 6.72, "densired_noise": 1999.90,
                     "magic_gamma": 0.30, "wine_quality": 3.88,
                     "pendigits": 4.04, "letter": 13.02}

    unsup_score = {"complex9": 35, "aggregation": 78, "har": 5,
                     "isolet": 3, "densired": 115, "densired_noise": 114,
                     "magic_gamma": 30, "wine_quality": -14,
                     "pendigits": 31, "letter": -1}

    bops = {}
    nn_dist = {}
    nn_acc = {}
    unsup = {}
    imb_score = {}

    for ds in dss:
        for sampling in samplings:
            for size in sizes:
                subset_log_name = f'subset_eval/{ds}_{sampling}_{size}.txt'
                subset_log_file = open(subset_log_name, 'r', buffering=1)
                lines = subset_log_file.readlines()
                subset_log_file.close()
                for line in lines:
                    line = line.strip("\n")
                    line = line.split(":")
                    mean = line[1].split(";")[0]
                    std = line[1].split(";")[1]
                    comb = [float(mean), float(std)]
                    key = f"{ds}_{sampling}_{size}"
                    if line[0] == "nn_acc":
                        nn_acc[key] = comb
                    elif line[0] == "nn_dist":
                        nn_dist[key] = comb
                    elif line[0] == "imb":
                        imb_score[key] = comb
                    elif line[0] == "unsup_score":
                        unsup[key] = comb
                    elif line[0] == "bop_js":
                        bops[key] = comb
    table_base_pre = "\\begin{table}[!tb]\n\\begin{center}\n\\caption{"
    table_base_mid = " of subsets of the datasets obtained through different sampling strategies ("
    table_base_post = " is better)}\\begin{tabular}{|l|c|c|c|c|c|}\\hline\nDataset & 1\% & 10\% & 25\%& 50\% & 75\%\\\\\\hline\n"
    table_base_post_100 = " is better, $\\times$ 100)}\\begin{tabular}{|l|c|c|c|c|c|}\\hline\nDataset & 1\% & 10\% & 25\%& 50\% & 75\%\\\\\\hline\n"
    table_bop = table_base_pre + "BoP Scores" + table_base_mid + "lower" + table_base_post_100
    table_nn_acc = table_base_pre + "Nearest Neighbor Classifier Accuracy" + table_base_mid + "higher" + table_base_post_100
    table_nn_dist = table_base_pre + "Nearest Neighbor Distance" + table_base_mid + "lower" + table_base_post
    table_unsup = table_base_pre + "Unsupervised Score" + table_base_mid + "higher is better)}\\begin{tabular}{|l|c|c|c|c|c|c|}\\hline\nDataset & 1\% & 10\% & 25\%& 50\% & 75\% & 100\%\\\\\\hline\n"
    table_imb = table_base_pre + "Imbalance Degree" + table_base_mid + "full dataset values included for reference)}\\begin{tabular}{|l|c|c|c|c|c|c|}\\hline\nDataset & 1\% & 10\% & 25\%& 50\% & 75\% & 100\%\\\\\\hline\n"


    for ds in dss:
        table_front = datasets_name[ds]
        for i in ["0.01", "0.1", "0.25", "0.5", "0.75"]:
            table_front += "& "
        table_front2 = table_front + "\\\\\n"
        table_bop += table_front2
        table_nn_acc += table_front2
        table_nn_dist += table_front2
        table_unsup += table_front + "&\\\\\n"
        table_imb += table_front + "&\\\\\n"



        for sampling in samplings:
            table_samp = f"- {sampling_name[sampling]}"
            table_bop += table_samp
            table_nn_acc += table_samp
            table_nn_dist += table_samp
            table_unsup += table_samp
            table_imb += table_samp

            for size in sizes:
                key = f"{ds}_{sampling}_{size}"
                bop_mean, bop_std = bops[key]
                nn_dist_mean, nn_dist_std = nn_dist[key]
                nn_acc_mean, nn_acc_std = nn_acc[key]
                unsup_mean, unsup_std = unsup[key]
                imb_mean, imb_std = imb_score[key]
                # if sampling in []: # ["random", "lwc", "kcentroid"]:
                #     table_bop += f"&{bop_mean*100:.1f} $\pm$ {bop_std*100:.0f}"
                #     table_nn_dist += f"&{nn_dist_mean:.3f} $\pm$ {nn_dist_std:.3f}"
                #     table_nn_acc += f"&{nn_acc_mean*100:.1f} $\pm$ {nn_acc_std*100:.0f}"
                #     table_unsup += f"&{unsup_mean:.1f} $\pm$ {unsup_std:.0f}"
                #     table_imb += f"&{imb_mean:.1f} $\pm$ {imb_std:.0f}"
                # else:
                if bop_mean == 0:
                    bopcolor = "\\cellcolor{blue!50}"
                #elif math.log(bop_mean*10) > 0:
                #    strength = min(60,round(math.log(bop_mean*10)*15))
                #    bopcolor = "\\cellcolor{red!" + str(strength) + "}"
                else:
                    strength = min(60, round(bop_mean*100))
                    bopcolor = "\\cellcolor{red!" + str(strength) + "}"
                table_bop += f"&{bopcolor}{bop_mean*100:.2f}"

                strength = round(min(60, nn_dist_mean*15))
                nndistcolor = "\\cellcolor{red!" + str(strength) + "}"
                table_nn_dist += f"&{nndistcolor}{nn_dist_mean:.3f}"

                if nn_acc_mean > 0.5:
                    strength = round(nn_acc_mean*100-50)
                    nn_acc_color = "\\cellcolor{blue!" + str(strength) + "}"
                else:
                    strength = round(50-nn_acc_mean * 100)
                    nn_acc_color = "\\cellcolor{red!" + str(strength) + "}"

                table_nn_acc += f"&{nn_acc_color}{nn_acc_mean*100:.2f}"

                strength = min(60,abs(round((unsup_mean-unsup_score[ds])/(unsup_score[ds]) * 45)))
                unsup_color = "\\cellcolor{red!" + str(strength) + "}"
                if size < 1:
                    table_unsup += f"&{unsup_color}{abs(unsup_mean-unsup_score[ds]):.0f}"


                strength = round(abs(imb_mean-base_id[ds])/class_num[ds] * 75)
                unsup_color = "\\cellcolor{purple!" + str(strength) + "}"
                if ds == "densired_noise":
                    table_imb += f"&{unsup_color}{imb_mean:.1f}"
                else:
                    table_imb += f"&{unsup_color}{imb_mean:.2f}"
            table_unsup += f"&{unsup_score[ds]:.0f}"

            if ds == "densired_noise":
                table_imb += f"&{base_id[ds]:.1f}"
            else:
                table_imb += f"&{base_id[ds]:.2f}"
            table_bop += "\\\\\n"
            table_nn_dist += "\\\\\n"
            table_nn_acc += "\\\\\n"
            table_unsup += "\\\\\n"
            table_imb += "\\\\\n"
        table_bop += "\\hline\n"
        table_nn_dist += "\\hline\n"
        table_nn_acc += "\\hline\n"
        table_unsup += "\\hline\n"
        table_imb += "\\hline\n"

    table_end_pre = "\\end{tabular}\n\\label{tab:"
    table_end_post = "}\n\\end{center}\n\\end{table}"
    table_bop += table_end_pre + "subset_bop" + table_end_post
    table_nn_acc += table_end_pre + "subset_nnacc" + table_end_post
    table_nn_dist += table_end_pre + "subset_nndist" + table_end_post
    table_unsup += table_end_pre + "subset_unsup" + table_end_post
    table_imb += table_end_pre + "subset_imb" + table_end_post

    print(table_bop)
    #print(table_nn_acc)
    print(table_nn_dist)
    print(table_unsup)
    #print(table_imb)

if __name__ == "__main__":
    make_sub_table()
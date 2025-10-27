import math
import os
import warnings

import numpy as np

warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)


def make_diff_table(alg, supervision, samplings, scaling1, scaling2):
#scaling1 - scaling2
    datasets = ["aggregation", "complex9", "densired", "densired_noise", "wine_quality", "isolet", "har", "pendigits", "magic_gamma", "letter"]
    datasets_name = {"complex9": "Complex-9", "aggregation": "Aggregation", "har": "HAR",
                     "isolet": "Isolet", "densired": "DENSIRED", "densired_noise": "DENSIRED$_{N}$",
                     "magic_gamma": "Magic-Gamma", "wine_quality": "Wine-Quality",
                     "pendigits": "Pendigits", "letter": "Letter"
                     }

    supervision_name = {"True": "supervised score (ARI + AMI)", "False": "unsupervised score (Silhouette Coefficient + DISCO)"}
    supervision_string = {"True": "supervised", "False": "unsupervised"}

    scaling_name = {"sample_mult": "linear scaling", "none": "no scaling", "root_unsup": "root scaling"}

    sampling_name = {"random":"Random", "lwc":"LWC", "protras": "ProTraS", "kcentroid": "k-Centroid", "dendis": "DenDis", "dides": "DiDes"}

    alg_name = {"dbscan": "DBSCAN", "kmeans": "k-Means", "spectral": "Spectral Clustering", "em": "Expectation–Maximization", "agglomerative": "Agglomerative Clustering", "dpc": "Density Peak Clustering", "hdbscan": "HDBSCAN"}

    table = "\\begin{table}[!tb]\n\\begin{center}\n"
    table += "\\caption{Difference in performance of " + alg_name[alg]
    table += " according to " + supervision_name[supervision]
    table += " for optimization on subset sizes generated with "
    table += sampling_name[samplings[0]]
    table += " subsampling between "
    table += scaling_name[scaling1]
    table += " and "
    table += scaling_name[scaling2]
    table += f". Scaled by 100, negative/red means {scaling_name[scaling2]} has larger values, positive/blue means {scaling_name[scaling1]} has larger values"
    table += ".}\n"

    table += "\\begin{tabular}{|l|c|c|c|c|}\\hline\n"
    table += "Dataset & 1\% & 10\% & 25\%& 50\%\\\\\\hline\n"

    for dataset in datasets:
        table += datasets_name[dataset]
        for sampling in samplings:#, "lwc", "kcentroid", "protras", "dendis", "dides"]:
            #table += f"- {sampling_name[sampling]}"
            prefix = f"eval_logs/{dataset}_{alg}/log_{dataset}_{alg}_{supervision_string[supervision]}_from_"
            postfix = f"_on_kcentroid_1.0_3600"
            for i in ["0.01", "0.1", "0.25", "0.5"]:#, "0.75"]:
                table += "& "
                path_scale_1 = f"{prefix}{sampling}_{i}{postfix}_{scaling1}.txt"
                path_scale_2 = f"{prefix}{sampling}_{i}{postfix}_{scaling2}.txt"
                subnum_str = ""
                if os.path.exists(path_scale_1) and os.path.exists(path_scale_2):
                    score1 = None
                    score2 = None
                    with open(path_scale_1) as file1:
                        lines = file1.readlines()
                        if len(lines) > 0:
                            scoring = lines[-1].replace("\n", "")
                            scoringsplit = scoring.split(" ")

                            if scoringsplit[-1].replace('.','',1).isnumeric():
                                subnum = round(float(scoringsplit[-1]))
                                if subnum > 1000000:
                                    subnum_str = f"{subnum / 1000000.0:.1f}k"
                                elif subnum > 1000:
                                    subnum_str = f"{subnum / 1000.0:.1f}k"
                                else:
                                    subnum_str = f"{subnum}"
                            else:
                                subnum_str = "-"

                            if scoringsplit[-4].replace('e','',1).replace('.','',1).replace('-','',2).isnumeric():
                                score1 = round((2-float(scoringsplit[-4]))*100)
                            elif scoringsplit[-4] == "inf":
                                score1 = np.inf
                            else:
                                score1 = None
                    with open(path_scale_2) as file2:
                        lines = file2.readlines()
                        if len(lines) > 0:
                            scoring = lines[-1].replace("\n", "")
                            scoringsplit = scoring.split(" ")
                            if scoringsplit[-4].replace('e','',1).replace('.','',1).replace('-','',2).isnumeric():
                                score2 = round((2-float(scoringsplit[-4]))*100)
                            elif scoringsplit[-4] == "inf":
                                score2 = np.inf
                            else:
                                score2 = None
                    if score1 is not None and score2 is not None:
                        if score2 == 0:
                            score2 = 0.000000000001
                        if score1 <= score2:
                            strength = abs((score1-score2) / score2)
                            table += "\\cellcolor{red!"
                            table += f"{min(50, strength * 50):.0f}"
                            table += "}"
                        else:
                            strength = abs((score1 - score2) / score2)
                            table += "\\cellcolor{blue!"
                            table += f"{min(50, strength * 50):.0f}"
                            table += "}"
                        table += f"{score1-score2:.0f} ({subnum_str})"
                    elif score1 is None and score2 is None:
                        table += "-"
                    elif score1 is None:
                        #table += "\\cellcolor{red!50}"
                        #table += f"{-score2:.0f} ({subnum_str})"
                        table += "-"
                    else:
                        #table += "\\cellcolor{blue!50}"
                        #table += f"{score1:.0f} ({subnum_str})"
                        table += "-"
                else:
                    table += "-"
            table += "\\\\\n"
        table += "\\hline\n"


    table += "\\end{tabular}\n"
    table += "\\label{tab:diff_" + f"{alg}" + "_" + f"{supervision_string[supervision]}"  + "_" + str(scaling1) + "-"+ str(scaling2)
    table += "}\n\\end{center}\n\\end{table}"
    print(table)

if __name__ == '__main__':
    make_diff_table("kmeans", "False", ["protras"], "root_unsup", "sample_mult")
    #make_diff_table("em", "False", ["protras"], "root_unsup", "sample_mult")
    #make_diff_table("spectral", "False", ["protras"], "root_unsup", "sample_mult")

    #make_diff_table("spectral", "True", ["random"], "none", "sample_mult")

    #make_diff_table("agglomerative", "True", ["random"], "none", "sample_mult")
    #make_diff_table("agglomerative", "False", ["protras"], "none", "sample_mult")

    make_diff_table("dbscan", "True", ["random"], "none", "sample_mult")
    #make_diff_table("dbscan", "False", ["protras"], "none", "sample_mult")

    #make_diff_table("spectral", "True", ["random"], "none", "sample_mult")
    #make_diff_table("spectral", "False", ["protras"], "root_unsup", "sample_mult")

    #make_diff_table("em", "False", ["protras"], "root_unsup", "sample_mult")
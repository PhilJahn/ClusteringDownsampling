import math
import os
import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)

# makes table for clustering algorithms given supervision state, sampling method and scaling approach
def make_table(alg, supervision, samplings, scaling):

    datasets = ["aggregation", "complex9", "densired", "densired_noise", "wine_quality", "isolet", "har", "pendigits", "magic_gamma", "letter"]
    datasets_name = {"complex9": "Complex-9", "aggregation": "Aggregation", "har": "HAR",
                     "isolet": "Isolet", "densired": "DENSIRED", "densired_noise": "DENSIRED$_{N}$",
                     "magic_gamma": "Magic-Gamma", "wine_quality": "Wine-Quality",
                     "pendigits": "Pendigits", "letter": "Letter"
                     }

    supervision_name = {"True": "supervised score (ARI + AMI)", "False": "unsupervised score (Silhouette Coefficient + DISCO)"}
    supervision_string = {"True": "supervised", "False": "unsupervised"}

    sampling_name = {"random":"Random", "lwc":"LWC", "protras": "ProTraS", "kcentroid": "k-Centroid", "dendis": "DenDis", "dides": "DiDes"}

    alg_name = {"dbscan": "DBSCAN", "kmeans": "k-Means", "spectral": "Spectral Clustering", "em": "Expectation–Maximization", "agglomerative": "Agglomerative Clustering", "dpc": "Density Peak Clustering", "hdbscan": "HDBSCAN"}

    table = "\\begin{table*}[!tb]\n\\begin{center}\n"
    table += "\\caption{Performance of " + alg_name[alg]
    table += " according to " + supervision_name[supervision]
    table += " for optimization on subset sizes generated with "
    table += sampling_name[samplings[0]]
    table += " subsampling. Scaled by 100, higher is better. Standard deviation is given. The number in brackets is the number of runs during optimization.}\n"
    table += "\\begin{tabular}{|l|c|c|c|c|c||c|c|}\\hline\n"
    table += "Dataset & 1\% & 10\% & 25\%& 50\% & 75\% & 100\% & 100\% (12 hours) \\\\\\hline\n"

    for dataset in datasets:
        table += datasets_name[dataset]
        #for i in ["0.01", "0.1", "0.25", "0.5", "0.75", "1.0", "1.0_2"]:
        #    table += "& "
        #table += "\\\\\n"
        for sampling in samplings:#, "lwc", "kcentroid", "protras", "dendis", "dides"]:
            #table += f"- {sampling_name[sampling]}"
            prefix = f"eval_logs/{dataset}_{alg}/log_{dataset}_{alg}_{supervision_string[supervision]}_from_"
            postfix = f"_on_kcentroid_1.0_3600"
            full_score = 200
            if os.path.exists(f"{prefix}random_1.0{postfix}_sample_mult.txt"):
                with open(f"{prefix}random_1.0{postfix}_sample_mult.txt", "r") as file:
                    lines = file.readlines()
                    if len(lines) > 0:
                        scoring = lines[-1].replace("\n", "")
                        scoringsplit = scoring.split(" ")
                        try:
                            full_score = round((2-float(scoringsplit[-4]))*100)
                        except:
                            full_score = -1
                    else:
                        full_score = -1
            else:
                full_score = -1
            long_score = 200
            if os.path.exists(f"{prefix}random_1.0_on_kcentroid_1.0_43200_sample_mult.txt"):
                with open(f"{prefix}random_1.0_on_kcentroid_1.0_43200_sample_mult.txt", "r") as file:
                    lines = file.readlines()
                    if len(lines) > 0:
                        scoring = lines[-1].replace("\n", "")
                        scoringsplit = scoring.split(" ")
                        try:
                            long_score = round((2-float(scoringsplit[-4]))*100)
                        except:
                            long_score = -1
                    else:
                        long_score = -1
            else:
                long_score = -1
            for i in ["0.01", "0.1", "0.25", "0.5", "0.75", "1.0", "1.0_2"]:
                table += "& "
                if i == "1.0_2":
                    path = f"{prefix}random_1.0_on_kcentroid_1.0_43200_sample_mult.txt"
                elif i == "1.0":
                    path = f"{prefix}random_1.0{postfix}_sample_mult.txt"
                else:
                    path = f"{prefix}{sampling}_{i}{postfix}_{scaling}.txt"
                if os.path.exists(path):
                    with open(path) as file:
                        lines = file.readlines()
                        if len(lines) > 0:
                            scoring = lines[-1].replace("\n", "")
                            scoringsplit = scoring.split(" ")
                            #https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-represents-a-number-float-or-int
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
                            #print(scoringsplit[-4], scoringsplit[-4].replace('.','',1).replace('-','',1).isnumeric())
                            if scoringsplit[-4].replace('e','',1).replace('.','',1).replace('-','',2).isnumeric():
                                subscore = round((2-float(scoringsplit[-4]))*100)
                                subdev = round(float(scoringsplit[-2])*100)
                                if i not in ["1.0_2", "1.0"] and full_score >= 0:
                                    if full_score == 0:
                                        full_score = 0.00000000000000001
                                    if subscore > long_score and subscore > full_score:
                                        table += "\\cellcolor{blue!65}"
                                    elif subscore < 0:
                                        table += "\\cellcolor{red!50}"
                                    elif full_score <= subscore:
                                        if long_score >= full_score:
                                            improvement_long = long_score - full_score
                                            improvement_sub = subscore - full_score
                                            if improvement_long > 0:
                                                strength = improvement_sub/improvement_long
                                                table += "\\cellcolor{blue!"
                                                table += f"{(strength * 50):.0f}"
                                                table += "}"
                                    else:
                                        strength = 1-(subscore/full_score)
                                        table += "\\cellcolor{red!"
                                        table += f"{(strength*50):.0f}"
                                        table += "}"
                                elif i not in ["1.0_2", "1.0"] and full_score < 0:
                                    if subscore > long_score and subscore > full_score:
                                        table += "\\cellcolor{blue!65}"
                                    elif full_score <= subscore:
                                        if long_score >= full_score:
                                            improvement_long = long_score - full_score
                                            improvement_sub = subscore - full_score
                                            if improvement_long > 0:
                                                strength = improvement_sub/improvement_long
                                                table += "\\cellcolor{blue!"
                                                table += f"{(strength * 50):.0f}"
                                                table += "}"
                                    else:
                                        strength = 1-(full_score/subscore)
                                        table += "\\cellcolor{red!"
                                        table += f"{(strength*50):.0f}"
                                        table += "}"
                                table += f"{subscore} $\\pm$ {subdev} ({subnum_str}) "
                            elif scoringsplit[-4] == "inf":
                                table += f"\\inf $\\pm$ 0 ({subnum_str}) "
                            else:
                                table += "-"
                        else:
                            table += "-"
                else:
                    table += "-"
            table += "\\\\\n"
        table += "\\hline\n"


    table += "\\end{tabular}\n"
    table += "\\label{tab:" + f"{alg}" + "_" + f"{supervision_string[supervision]}"  + "_" + str(scaling)
    table += "}\n\\end{center}\n\\end{table*}"
    print(table)

# prints tables for evaluations to console, only content is used, headers were reused from prior state and not upated here
if __name__ == '__main__':
    make_table("kmeans", "True", ["random"], "sample_mult")
    make_table("kmeans", "False", ["protras"], "sample_mult")
    #make_table("kmeans", "False", ["protras"], "root_unsup")
    print("-----")
    make_table("em", "True", ["random"], "sample_mult")
    make_table("em", "False", ["protras"], "sample_mult")
    #make_table("em", "False", ["protras"], "root_unsup")
    print("-----")
    make_table("spectral", "True", ["random"], "sample_mult")
    make_table("spectral", "False", ["protras"], "sample_mult")
    #make_table("spectral", "False", ["protras"], "root_unsup")
    print("-----")
    make_table("dbscan", "True", ["random"], "sample_mult")
    make_table("dbscan", "False", ["protras"], "sample_mult")
    print("-----")
    make_table("agglomerative", "True", ["random"], "sample_mult")
    make_table("agglomerative", "False", ["protras"], "sample_mult")
    print("-----")
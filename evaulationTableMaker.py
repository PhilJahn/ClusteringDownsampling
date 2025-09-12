import math
import os
import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.warn("userwarning", UserWarning)
warnings.warn("runtimewarning", RuntimeWarning)


def make_table(alg, supervision):

    datasets = ["complex9", "aggregation"]
    datasets_name = {"complex9": "Complex-9", "aggregation": "Aggregation"}

    supervision_name = {"True": "supervised score (ARI + AMI)"}
    supervision_string = {"True": "supervised", "False": "unsupervised"}

    sampling_name = {"random":"Random", "lwc":"LWC", "protras": "ProTraS", "kcentroid": "k-Centroid"}

    alg_name = {"dbscan": "DBSCAN", "kmeans": "k-Means", "spectral": "Spectral Clustering", "em": "Expectation–Maximization", "agglomerative": "Agglomerative Clustering", "dpc": "Density Peak Clustering", "hdbscan": "HDBSCAN"}

    table = "\\begin{table*}\n\\begin{center}\n"
    table += "\\caption{Performance of " + alg_name[alg]
    table += " according to " + supervision_name[supervision]
    table += " for optimization on subset sizes. Scaled by 100, higher is better. Standard deviation is given. The number in brackets is the number of runs during optimization.}\n"
    table += "\\begin{tabular}{|l|c|c|c|c|c||c|c|}\\hline\n"
    table += "Dataset & 1\% & 10\% & 25\%& 50\% & 75\% & 100\% & 100\% (12 hours) \\\\\\hline\n"

    for dataset in datasets:
        table += datasets_name[dataset]
        for i in ["0.01", "0.1", "0.25", "0.5", "0.75", "1.0", "1.0_2"]:
            table += "& "
        table += "\\\\\n"
        for sampling in ["random", "lwc", "kcentroid", "protras"]:
            table += f"- {sampling_name[sampling]}"
            prefix = f"eval_logs/{dataset}_{alg}/log_{dataset}_{alg}_{supervision_string[supervision]}_from_"
            postfix = f"_on_kcentroid_1.0_3600_sample_mult.txt"
            full_score = 200
            if os.path.exists(f"{prefix}random_1.0{postfix}"):
                with open(f"{prefix}random_1.0{postfix}", "r") as file:
                    lines = file.readlines()
                    scoring = lines[-1].replace("\n", "")
                    scoringsplit = scoring.split(" ")
                    full_score = round((2-float(scoringsplit[-4]))*100)
            for i in ["0.01", "0.1", "0.25", "0.5", "0.75", "1.0", "1.0_2"]:
                table += "& "
                if i == "1.0_2":
                    path = f"{prefix}random_1.0_on_kcentroid_1.0_43200_sample_mult.txt"
                elif i == "1.0":
                    path = f"{prefix}random_1.0{postfix}"
                else:
                    path = f"{prefix}{sampling}_{i}{postfix}"
                if os.path.exists(path):
                    with open(path) as file:
                        lines = file.readlines()
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
                            if i not in ["1.0_2", "1.0"]:
                                if full_score >= subscore:
                                    table += "\\cellcolor{red!"
                                    table += f"{(((full_score-subscore)/full_score)*50):.0f}"
                                    table += "}"
                                else:
                                    table += "\\cellcolor{blue!"
                                    table += f"{min(50,((subscore-full_score)/full_score)*100):.0f}"
                                    table += "}"
                            table += f"{subscore} $\\pm$ {subdev} ({subnum_str}) "
                        elif scoringsplit[-4] == "inf":
                            table += f"0 $\\pm$ 0 ({subnum_str}) "
                        else:
                            table += "-"
                else:
                    table += "-"
            table += "\\\\\n"
        table += "\\hline\n"


    table += "\\end{tabular}\n"
    table += "\\label{tab:" + f"{alg}"
    table += "}\n\\end{center}\n\\end{table*}"
    print(table)

if __name__ == '__main__':
    for alg in ["dbscan", "hdbscan", "kmeans", "spectral", "dpc", "agglomerative", "em"]:
        make_table(alg, "True")


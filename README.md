This repository contains the code for a paper submitted to ICDE 2026.  

## Structure

This repository is made up of several subfolders storing the artifacts of different steps of the paper.
* /data contains all datasets*, the datasets are either self-generated (/data/synth/), from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)  (/data/uci/ and data/uci_download/) or from [milaan9's clustering data repository](https://github.com/milaan9/Clustering-Datasets) (/data/uci_milaan9/ and /data/synthetic_milaan9/).
  Aside from these /data contains all downsampled datasets with the folder name corresponding to the used downsampling strategy
* /opt_logs contains the logged information on all tested configurations during optimization. The log files are sorted into folders "dataset_clustering-algorithm" and all files are named according to "log_dataset_clustering-algorithm_supervision_downsampling-method_downsampling-size_optimization-budget_data-seed_smac-seed.txt". As this folder is very large, it is stored on Zenodo, rather than on GitHub.
* /param_logs contains the information on the final configurations obtained from optimization. The files are sorted into folders "dataset_clustering-algorithm" and all files are named according to "log_dataset_clustering-algorithm_supervision_downsampling-method_downsampling-size_optimization-budget.csv".
* /eval_logs contains the evaluations of the obtained hyperparameter configurations on relevant datasets. The files are sorted into folders "dataset_clustering-algorithm" and all files are named according to "log_dataset_clustering-algorithm_supervision_from_downsampling-method-1_downsampling-size-1_to_downsampling-method-2_downsampling-size-2_optimization-budget_scaling-method.csv".
* /bop contains the artifacts produced by the Bag of Prototypes calculation for subset evaluation. The structure is "dataset/downsampling-method/downsampling-size/data-seed".
  
(* The train data of HAR is too big for non-LFS GitHub and, as such, is stored on [Zenodo](TODO). Alternatively, the dataset can be downloaded [here](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).)

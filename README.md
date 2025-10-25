This repository contains the code for a paper submitted to ICDE 2026.  

## Datasets

To start with, datasets are required for running any experiments:
* ```/data/``` contains all datasets*, the datasets are either self-generated (```/data/synth/```), from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)  (```/data/uci/``` and ```data/uci_download/```) or from [milaan9's clustering data repository](https://github.com/milaan9/Clustering-Datasets) (```/data/uci_milaan9/``` and ```/data/synthetic_milaan9/```).
* ```/data_eval/``` contains the properties of all datasets in the files with the corresponding name, created using ```data_eval.py``` (parameter: --ds: dataset to analyze)
* ```make_densired_data.py``` was used to generate datasets using the [densired](https://github.com/PhilJahn/DENSIRED/) package
* ```data_handler.py``` manages the datasets.

All datasets are referenced through their keys:
| **Name**        | **Key** |
|----------------------|-------------|
| Aggregation | aggregation |
| Complex-9 | complex9 |
| DENSIRED | densired |
| DENSIRED_N | densired_noise |
| Scaling 1 | scaling1 |
| Scaling 2 | scaling2 |
| Aggregation 2 | aggregation2 |
| Complex-9 2 | complex92 |
| Wine-Quality | wine_quality |
| Isolet | isolet |
| HAR | har |
| Pendigits | pendigits |
| Magic-Gamma | magic_gamma |
| Letter | letter |
| Large | large |
| VeryLarge | verylarge3 |

(* The train data of HAR is too big for non-LFS GitHub and, as such, is stored on [Zenodo](TODO). Alternatively, the dataset can be downloaded [here](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).)

## Downsampling

Downsampling is performed using ```subset_handler.py```

| **Parameter**        |  **Function**  |
|----------------------|-------------------|
| --ds                      | Dataset to downsample |
| --size                      | Ratio/size to downsample to |
| --sampling                      | Downsampling method to use |
| --data_seed                      | Seed for random instantiation of downsampling strategy |

Here, the following keys correspond to the implemented downsampling strategies:

| **Name**        | **Key** |
|-----------------|----------- |
| Uniform Random Sampling | random |
| LWC | lwc |
| k-Centroid | kcentroid |
| DenDis | dendis |
| DisDen | disden |
| ProTraS | protras |

(Additional downsampling strategies from BIRCHSCAN and random instances from k-Means Clusters are implemented, but were not used)
* ```/data``` contains all downsampled datasets with the folder name corresponding to the used downsampling strategy
* ```/subset_eval``` contains the analysis of all subsets in the files with the corresponding name, created using ```data_eval.py``` (parameter: --ds: dataset, --size: downsampling size, --sampling: downsampling method). The files contain both the general properties of the data and the similarity to the full dataset. The file naming is ```dataset_downsampling-method_downsampling-size.txt```
* ```/bop``` contains the artifacts produced by the Bag of Prototypes calculation for subset evaluation. The structure is ```dataset/downsampling-method/downsampling-size/data-seed/```.

## Hyperparameter Optimization
* /opt_logs contains the logged information on all tested configurations during optimization. The log files are sorted into folders "dataset_clustering-algorithm" and all files are named according to "log_dataset_clustering-algorithm_supervision_downsampling-method_downsampling-size_optimization-budget_data-seed_smac-seed.txt". As this folder is very large, it is stored on Zenodo, rather than on GitHub.
* /param_logs contains the information on the final configurations obtained from optimization. The files are sorted into folders "dataset_clustering-algorithm" and all files are named according to "log_dataset_clustering-algorithm_supervision_downsampling-method_downsampling-size_optimization-budget.csv".
* /eval_logs contains the evaluations of the obtained hyperparameter configurations on relevant datasets. The files are sorted into folders "dataset_clustering-algorithm" and all files are named according to "log_dataset_clustering-algorithm_supervision_from_downsampling-method-1_downsampling-size-1_to_downsampling-method-2_downsampling-size-2_optimization-budget_scaling-method.csv"

## External Code

* The BoP code is in ```/similarity/BoP.py``` and originally comes from the [official implementation repository](https://github.com/Klaus-Tu/Bag-of-Prototypes).
* The DISCO code was provided by the original authors. The official GitHub repository will be linked here once it is up.


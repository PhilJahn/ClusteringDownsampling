This repository contains the code for a paper submitted to ICDE 2026.  

## Requirements

The framework is designed for Python 3.12 on Linux. Timeout checks for optimization are performed using SIGALRM, which is available only on Linux. 

The pip freeze output for our virtual environment is in ```requirements.txt```. It may be necessary to install ```swig``` and ```cmake``` before the other packages, as ```pyrfr```, which is required by SMAC3, can cause trouble otherwise.

DENSIRED requires ```numpy==2.0``` or higher, which is incompatible with some other packages that require earlier versions of ```numpy```, and is thus left out of ``````requirements.txt```. It is only needed if ```make_densired_data.py``` is run, otherwise the stored datasets are used.

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
| Scaling | scaling1 |
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
| VeryLarge-2 | verylarge |
| VeryLarge-3 | verylarge3 |

(* The train data of HAR is too big for non-LFS GitHub and, as such, is stored on [Zenodo](https://zenodo.org/records/17456047). Alternatively, the dataset can be downloaded [here](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).)

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
* ```/data/``` contains all downsampled datasets with the folder name corresponding to the used downsampling strategy
* ```/subset_eval/``` contains the analysis of all subsets in the files with the corresponding name, created using ```data_eval.py``` (parameter: --ds: dataset, --size: downsampling size, --sampling: downsampling method). The files contain both the general properties of the data and the similarity to the full dataset. The file naming is ```dataset_downsampling-method_downsampling-size.txt```
* ```/bop/``` contains the artifacts produced by the Bag of Prototypes calculation for subset evaluation. The structure is ```dataset/downsampling-method/downsampling-size/data-seed/```.

## Hyperparameter Optimization

The optimization using [SMAC3](https://github.com/automl/SMAC3) is performed using ```findParameters.py```


| **Parameter**        |  **Function**  |
|----------------------|-------------------|
| --ds                      | Dataset |
| --size                      | Downsampling size|
| --sampling                      | Downsampling method |
| --method           | Clustering method to optimize |
| --budget | Optimization time budget in seconds |
| --data_seed         | Seed for downsampling (-1 iterates over 0,1,2 for non-deterministic methods, each requiring time corresponding ```--budget```) |
| --smac_seed         | Seed for SMAC§ (-1 iterates over 0,1,2, each requiring time corresponding ```--budget``` (this stacks with  ```--data_seed ```) |
| --supervised | Whether to use the supervised or unsupervised score (Boolean as Integer, 1=Supervised, 0=Unsupervised) |
| --overwrite | Whether to overwrite existing optimization runs (Boolean as Integer, 1=Yes, 0=No) |

Clustering methods follow this naming scheme. All use their  [Scikit-Learn](https://scikit-learn.org) implementations.

| **Name**        | **Key** |
|-----------------|----------- |
| k-Means | kmeans |
| Expectation Maximization with GMM | em |
| Spectral Clustering | spectral |
| DBSCAN | dbscan |
| Agglomerative | agglomerative |

(Additional clustering strategies, HDBSCAN (also Scikit-Learn) and DPC (self-implemented), are implemented, but were not used in the paper)
* ```/opt_logs/``` contains the logged information on all tested configurations during optimization. The log files are sorted into folders ```/opt_logs/dataset_clustering-algorithm/``` and all files are named like this: ```log_dataset_clustering-algorithm_supervision_downsampling-method_downsampling-size_optimization-budget_data-seed_smac-seed.txt```. As this folder is very large, it is stored on [Zenodo](https://zenodo.org/records/17456047), rather than on GitHub.
* ```/param_logs/``` contains the information on the final configurations obtained from optimization. The files are sorted into folders ```/param_logs/dataset_clustering-algorithm``` and all files are named like this: ```log_dataset_clustering-algorithm_supervision_downsampling-method_downsampling-size_optimization-budget.csv```.

The configuration spaces for this are handled through ```config_handler.py```. If a new dataset is added, it needs to receive an entry in the ```class_num```-dictionary to function (as downsampled datasets may drop class labels, but the default hyperparameters should stay the same).

The clustering algorithms are handled through ```clustering_handler.py```.

## Scaling Behavior Analysis

The hyperparameter scaling analysis is a two-step process. 
First, a grid search across has to be performed for every dataset that is supposed to be compared (our used keys were: ```scaling1```, ```scaling2```, ```agglomerative```, ```agglomerative2```, ```complex9```, ```complex92```). The grid search is performed using ```evaluateParamterGrid.py```. It supports downsampled/upsampled datasets, but ultimately was doen using fixed datasets.


| **Parameter**        |  **Function**  |
|----------------------|-------------------|
| --ds                      | Dataset |
| --size                      | Downsampling size|
| --sampling                      | Downsampling method |
| --method           | Clustering method |
| --param_configs | Number of configurations to test |
| --primary | First Hyperparameter to iterate over |
| --secondary | Secondary Hyperparameter to iterate over |
| --tertiary | Third Hyperparameter to iterate over (in case of Spectral Clustering, the ```assign_labels``` value can be swapped to ```discretize``` with ```assign_labels_d``` and to ```cluster_qr``` with ```assign_labels_c```) |

* ```/grid_evals/``` contains the results for this. They follow the naming scheme of ```dataset_downsampling-method_downsampling-size_clustering-algorithm_primary_secondary_tertiary_config-number.txt```. Each line corresponds to an evaluated run. Note: We only include completed results here, as incomplete results could still be processed by the later steps, which could lead to skewed results.

We used the following configurations:
* ```--method kmeans --primary n_clusters --secondary init --tertiary none```
* ```--method dbscan --primary eps --secondary min_samples --tertiary none```
* ```--method em --primary n_components --secondary init_params --tertiary covariance_type```
* ```--method spectral --primary n_clusters --secondary gamma --tertiary none```
* ```--method spectral --primary n_clusters --secondary n_neighbors --tertiary none```
* ```--method spectral --primary n_clusters --secondary gamma --tertiary assign_labels_d```
* ```--method spectral --primary n_clusters --secondary n_neighbors --tertiary assign_labels_d```
* ```--method spectral --primary n_clusters --secondary gamma --tertiary assign_labels_c```
* ```--method spectral --primary n_clusters --secondary n_neighbors --tertiary assign_labels_c```
* ```--method agglomerative --primary n_clusters --secondary linkage --tertiary none```
* ```--method agglomerative --primary n_clusters --secondary n_neighbors --tertiary linkage```

To extract information on the scaling behavior, we use ```scaling_eval.py```

| **Parameter**        |  **Function**  |
|---------------------------|-------------------|
| --up                      | Larger Dataset |
| --reg                      | Smaller Dataset |
| --factor                      | Upsampling ratio |
| --metric           | Metric to use for the distance function (sup for supervised score, unsup for unsupervised score) |
| --rebuild | Whether to rebuild the extraction dictionaries of the grid searches (Boolean as Integer, 1=Supervised, 0=Unsupervised) |
| --method           | Clustering method (Spectral Clustering using ```affinity```:```rbf``` is ```spectral_gamma``` and using ```affinity```:```nearest_neighbors``` is ```spectral_nn```, Agglomerative Clustering using ```connectivity```:```None``` is ```agglomerative_unstructured``` and using ```connectivity```:```kneighbors_graph``` is ```agglomerative```)|
| --submethod           | Clustering submethod (typically ```none```, but supports each of the ```linkage``` types of Agglomerative Clustering (```ward```, ```average```, ```single```, ```complete```)|

* ```/grid_dicts/``` contains all extraction dictionaries. The naming scheme is  ```dataset_method_metric.npy```.
* ```/scale_logs/``` contains all extracted scaling bahviors. The naming scheme is ```reg_to_up_method_submethod_metric.txt```.
* 
## Configuration Evaluation

To evaluate the obtained hyperparameter configurations, they are rerun to get their performance on (potentially) different dataset sizes.

| **Parameter**        |  **Function**  |
|----------------------|-------------------|
| --ds                      | Dataset |
| --size                      | Downsampling size of original (downsampled) dataset|
| --sampling                      | Downsampling method of original (downsampled) dataset |
| --evalsize                      | Downsampling size of evaluation (typical full) datatset|
| --sampling                      | Downsampling method of evaluation (typical full) datatset (not used for size greater or equal to 1) |
| --method           | Clustering method to optimize |
| --budget | Optimization time budget in seconds |
| --supervised | Whether to use the supervised or unsupervised score (Boolean as Integer, 1=Supervised, 0=Unsupervised) |
| --scaling | Which scaling behavior to apply |

The scaling behaviors are bundled as follows. Hyperparameters not listed here are unaffected by scaling (i.e. if no hyperparameters of an algorithm are included, the scaling option is equivalent to no scaling):
* ```none``` applies no scaling to any hyperparameter
* ```sample_mult``` applies linear scaling to ```min_samples``` (DBSCAN), ```n_neighbors``` (Agglomerative + Spectral Clustering) and ```min_cluster_size``` (unused)
* ```sample_root``` applies root scaling to ```min_samples``` (DBSCAN), ```n_neighbors``` (Agglomerative + Spectral Clustering) and ```min_cluster_size``` (unused)
* ```sample_square``` applies quadratic scaling to ```min_samples``` (DBSCAN), ```n_neighbors``` (Agglomerative + Spectral Clustering) and ```min_cluster_size``` (unused)
* ```root_unsup``` applies linear scaling to ```min_samples``` (DBSCAN), ```n_neighbors``` (Agglomerative + Spectral Clustering) and ```min_cluster_size``` (unused) and root scaling to ```n_clusters``` (k-Means + Agglomerative + Spectral Clustering) and ```n_components``` (EM)

Due to a bug, the scores in optimization are reported at 3/5 of their actual values. This does not affect candidate choice as it affects all scores universally, but for evaluation logs the subset optimization scores are reported lower than expected. The final scores are the actual performances of the clustering algorithms.

* /eval_logs contains the evaluations of the obtained hyperparameter configurations on relevant datasets. The files are sorted into folders "dataset_clustering-algorithm" and all files are named according to "log_dataset_clustering-algorithm_supervision_from_downsampling-method-1_downsampling-size-1_to_downsampling-method-2_downsampling-size-2_optimization-budget_scaling-method.csv"

## Paper Tools

To facilitate the information extraction from the gathered data in order to put it into the paper, we added several tools to automate the process.
* ```evaluationTableMaker.py``` produces tables for the final results
* ```evaluationTableMaker.py``` produces tables for the final results on the Large, VeryLarge-2 and VeryLarge-3 datasets
* ```differenceTableMaker.py``` produces tables comparing two scaling functions
* ```gridFigureMaker.py``` produces parameter grid figures in 1x4 shape (EM/Fig. 1)
* ```gridFigureMaker2.py``` produces parameter grid figures in 2x2 shape (DBSCAN/Fig. 2)
* ```legendMaker.py``` produces the legend for the heatmaps of Fig.1 and Fig.2
* ```subsetRanker.py``` computes the ranks of performance scores for downsampling methods
* ```subsetTableMaker.py``` produces tables for the metrics for the downsampling strategies

## External Code

* The BoP code is in ```/similarity/BoP.py``` and originally comes from the [official implementation repository](https://github.com/Klaus-Tu/Bag-of-Prototypes).
* The DISCO code was provided by the original authors. The official GitHub repository will be linked here once it is up.




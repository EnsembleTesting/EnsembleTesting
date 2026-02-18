# Ensemble-Based Test Selection for Fault Detection and Retraining in Deep Neural Networks
This repository provides the implementation of the paper 'Ensemble-Based Test Selection for Fault Detection and Retraining in Deep Neural Networks'.

## Abstract:  
Testing deep learning (DL)–based systems is essential for identifying internal defects and improving model robustness. Existing test selection metrics have demonstrated effectiveness by focusing on specific characteristics, such as model uncertainty or input diversity. However, these metrics are typically applied in isolation, limiting their ability to capture complementary testing signals. More importantly, existing ensemble-based approaches generally rely on a conditional selection strategy (e.g., apply different individual metrics based on conditions) or search-based trade-offs (e.g., use NSGA-II). We identify two key challenges in prior research:
(1) the absence of a statistically principled method for unifying and combining multiple test selection metrics, and
(2) the lack of a comprehensive evaluation of metrics under diverse distribution shifts and data types. To address these challenges, we propose an ensemble-based test selection method that introduces a statistically grounded scale unification mechanism to integrate multiple selection metrics. To determine which metrics should be integrated into our ensemble, we conduct a comprehensive empirical study evaluating existing metrics using two complementary criteria: the capability of fault detection and retraining guidance, and select the top-performing metrics for integration. 

We evaluate the proposed ensemble-based selection with two criteria across two Android malware datasets and two real-world autonomous driving datasets. Experimental results show that our method consistently outperforms both individual metrics and representative ensemble baselines, achieving statistically significant improvements under a wide range of out-of-distribution (OOD) scenarios. An ablation study further validates the contributions of the proposed scale unification and ensemble-based selection. To support reproducibility and future research, our implementation is publicly available at GitHub.

## Hyperparameters.
| Methods      | Param.   | Candidate values                     | Select | Value            |
|--------------|----------|--------------------------------------|--------|------------------|
| DeepDrebin   | layer    | [dense1, dense2]                     | GS     | dense1           |
| BasicDNN     | layer    | [dense1, dense2]                     | GS     | dense1           |
| UMAP         | m        | [5, 10, 25, 50, 75, 100]              | GS     | 25               |
| UMAP         | d        | [0.01, 0.1, 0.3, 0.5]                 | GS     | 0.1              |
| UMAP         | k        | [5, 15, 25, 50]                      | GS     | 15               |
| PCA          | m        | [5, 10, 25, 50, 75, 100]              | GS     | 25               |
| GRP          | m        | [5, 10, 25, 50, 75, 100]              | GS     | 25               |
| DBSCAN       | Min_k    | [2, 5, 10, 15]                       | GS     | 2                |
| DBSCAN       | ε        | NA                                   | KP     | Varied           |
| HDBSCAN      | MinPts   | [2, 5, 10, 15]                       | GS     | 2                |
| HAC          | K        | range[5, 35]                         | KP     | [10, 10]         |
| K-Means      | K        | range[5, 35]                         | KP     | [9, 15]          |
| Retraining   | bs       | [128, 256, 512]                      | GS     | 128              |
| Retraining   | epochs   | [5, 10, 20, 30, 40]                  | GS     | 30               |
| Retraining   | lr       | [1e-2, 1e-3, 1e-4, 1e-5]              | GS     | [1e-3, 1e-4]     |
| Ours         | r        | [0.3, 0.5, 0.7, 1.0]                 | GS     | 0.7              |
| Ours         | n        | [30, 60, 120, 180]                   | GS     | 60               |

This table summarizes the hyperparameters considered in this study (hyperparameters related to clustering pipeline, retraining, and our method), including their candidate values, selection methods, and the final optimal values adopted in our experiments. layer indicates the layer for feature extraction, $m, d, k$ denote the number of components, minimum distance, and the number of neighbors, respectively. $Min\_k$ denotes the minimum number of neighbors, $\epsilon$ is the distance threshold. $MinPts$ is the pre-defined minimum cluster size. $K$ is the pre-defined number of clusters. $bs, epochs, lr$ denote the batch size, the number of epochs, and the learning rate, respectively. $r,n$ denote the ratio and number of selections in our method.  

For methods employing grid search (GS), the optimal value is selected based on the average of the Silhouette and DBCV scores. For methods relying on the knee-point strategy (KP), we provide detailed guidance on curve computation. Regarding feature extraction, both DeepDrebin and BasicDNN contain two hidden layers. We experimentally observe that shallow-layer features consistently yield higher Silhouette and DBCV scores. For dimensionality reduction using UMAP, we define the candidate range of $m$ such that $m \leq |X^{mis}_s|$, addressing the curse of dimensionality. This principle similarly applies to PCA and GRP. In practice, candidate values exceeding the number of mispredicted samples are filtered out. For parameters $d$ and $k$, we follow the recommended ranges from the official UMAP documentation (https://umap-learn.readthedocs.io/en/latest/parameters.html). The candidate values for $Min_k$ and $MinPts$ in DBSCAN and HDBSCAN are determined by referring to official documentation (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html,https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) and relevant prior work [1, 2]. For K-Means and HAC, we restrict the candidate range of $K$ to [5, 35] to reflect the relatively small number of mispredicted inputs from the original test set. The selected optimal values are $K=9$ for AndroZoo and $K=15$ for Drebin in K-Means, and $K=10$ for both datasets in HAC. Since the optimal $\epsilon$ varies across settings, we apply the knee-point method in each experiment to determine the best value. For other aforementioned hyperparameters in the clustering pipeline, we find that the optimal values remain consistent across settings and thus fix them as shown in the table in our main experiments.

For both the retraining procedure and our proposed method, hyperparameter values are selected via grid search based on initial exploratory experiments. Specifically, we determine the optimal batch size ($bs$) and learning rate ($lr$) by identifying the configuration that yields the greatest accuracy improvement, while the number of training epochs is selected based on convergence behavior. We adopt a learning rate of $1\text{e}^{-3}$ for malware data and $1\text{e}^{-4}$ for driving data across both the retraining and original training processes. In our method, $r$ represents the ratio of samples to retain after filtering from predefined candidate sets, and $n$ represents the number of random selections. Based on preliminary validation results, we fix r = 0.7 and n = 60 for all subsequent experiments. We observe that r = 0.7 reduces the search space while improving the suite quality. n \in \{120,180\} provides similar performance while incurring higher computational cost, while $n = 30$ leads to slight degradation. Therefore, we use $n = 60$ to balance effectiveness and computational efficiency.

# Environment:
To successfully run the code, we provide two conda environment files. <br />
-environmentdnn.yml is the file for the main experiments, except for computing the clusters.<br />
-environmentdetect.yml is the file for conducting clustering and computing the number of clusters with the given test suite.

Start by cloning the repository on your local folder and then, to install, just run:<br />
`conda env create -f environmentdnn.yml` <br />
`conda env create -f environmentdetect.yml` <br />

# Dataset:
## Driving data (Dave, Udacity):
For Dave, please follow the link to download:<br />
https://github.com/SullyChen/Autopilot-TensorFlow

For Udacity, please follow the official procedure to download (Ch2_002): <br />
https://github.com/udacity/self-driving-car/tree/master/datasets

## Android Packages (AndroZoo, Drebin):
For AndroZoo, please follow the procedure from the official website to download:<br />
https://androzoo.uni.lu/

For Drebin, we directly use the extracted feature, provided by the authors of the paper: <br />
Adversarial Deep Ensemble: Evasion Attacks and Defenses for Malware Detection (TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY), by Deqiang Li and Qianmu Li.

# Run:
## Test selection metrics to guide retraining:
`conda activate dnntest`

For malware data, <br />
`cd adv-dnn-ens-malware`<br />
`python retrain_selected.py` for adversarial OOD operators.<br />
`python retrain_selected_outofsrc.py` for out-of-src OOD operators.<br />

For driving data, <br />
`cd adv-self-driving`<br />
`python retrain_selected.py` for adversarial OOD operators.<br />
`python retrain_selected_outofsrc.py` for out-of-src OOD operators.<br />

Please specify the interested parameters (metric, budget, id_ratio, OOD method, epochs) in the script.

## Test selection metrics to detect faults:
`conda activate faultdetect`

For malware data, <br />
`cd adv-dnn-ens-malware`<br />
`python3 root_cause_detect_malware_gpu.py`<br />

For driving data, <br />
`cd adv-self-driving`<br />
`python3 root_cause_detect_gpu.py`<br />

Please specify the interested parameters (metric, budget, id_ratio, OOD method, dataset, model) in the script.

## References.
[1] Z. Aghababaeyan, M. Abdellatif, L. Briand, S. Ramesh, and M. Bagherzadeh, “Black-box testing of deep neural networks through test case diversity,” IEEE Transactions on Software Engineering, vol. 49, no. 5, pp. 3182–3204, 2023.

[2] M. O. Attaoui, H. Fahmy, F. Pastore, and L. Briand, “Supporting safety analysis of image-processing dnns through clustering-based approaches,” ACM Transactions on Software Engineering and Methodology, vol. 33, no. 5, pp. 1–48, 2024.

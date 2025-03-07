# Unified Ensemble-Based Testing for Fault Detection and Retraining in Deep Neural Networks
This repository contains experiments conducted in the paper 'Unified Ensemble-Based Testing for Fault Detection and Retraining in Deep Neural Networks', submitted to the IEEE Transactions on Software Engineering journal.

## Abstract:  
Testing deep learning (DL)-based systems is critical for identifying internal defects and improving their robustness. Existing test selection metrics have shown promising results and focus on individual aspects like uncertainty or neuron coverage but fail to consider consensus decisions across metrics, limiting fault detection and retraining effectiveness. Specifically, we observed three key limitations: 1. Lack of consideration for the consensus of results from multiple independent metrics (i.e., selecting inputs based on a fused score from multiple metrics); 2. Insufficient effectiveness evaluation; 3. Limited scope of tasks and datasets. 

In this paper, we propose an ensemble-based testing framework that integrates scale and granularity unification to systematically combine multiple test selection metrics, by examining the effectiveness of test selection metrics on both the capability of fault detection and the capability of retraining guidance, and assessing the performance on two different domains.
Experimented and validated on two Android malware and two real-world driving datasets, our method achieves statistically significant improvements in fault detection (e.g., detecting 23 faults on average) and retraining guidance (e.g., achieving an average of 34.04% accuracy improvement for malware datasets and 13.22% for driving datasets). Compared to state-of-the-art baselines, our framework offers superior testing reliability and robustness across diverse out-of-distribution (OOD) scenarios. Additionally, our ablation study statistically validates the effectiveness of each component of our method, including scale unification and ensemble-based selection. All code and data are open-source to support further research.

# Environment:
To successfully run the code, we provide two conda environment files. <br />
-environmentdnn.yml is the file for the main experiments except for computing the clusters.<br />
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

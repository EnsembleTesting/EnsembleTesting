# Unified Ensemble-Based Testing for Fault Detection and Retraining in Deep Neural Networks
This repository contains experiments conducted in the paper 'Unified Ensemble-Based Testing for Fault Detection and Retraining in Deep Neural Networks'.

## Abstract:  
Testing deep learning (DL)-based systems is essential for identifying internal defects and enhancing model robustness. While existing test selection metrics have shown promise by targeting specific aspects such as model uncertainty or input diversity, they often operate in isolation, overlooking the potential benefits of consensus-based decision-making. This limitation hampers both fault detection and retraining effectiveness. We identify three key challenges in prior work: (1) the lack of integrated consensus among multiple metrics (i.e., input selection based on fused multi-metric scores), (2) insufficient evaluation of metric effectiveness, and (3) limited generalizability across tasks and domains.

To address these gaps, we propose an ensemble-based testing framework that incorporates scale and granularity unification to systematically combine multiple test selection metrics. We evaluate the effectiveness of our approach across two dimensions, fault detection and retraining guidance, on two Android malware datasets and two real-world autonomous driving datasets. Compared to state-of-the-art baselines, our framework achieves statistically significant improvements under diverse out-of-distribution (OOD) scenarios, detecting an average of 35 faults and improving model accuracy by 24.45%. An ablation study further confirms the contributions of individual components, including scale unification and ensemble-based selection. To facilitate future research, we release all code as open source.

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

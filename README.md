#Ensemble-based Testing for Deep Neural Networks through Scale Unification
This repository contains experiments conducted in the paper 'Ensemble-based Testing for Deep Neural Networks through Scale Unification', submitted to IEEE Transactions on Software Engineering journal
##Abstract:  Testing Deep learning (DL)-based systems is a critical step in identifying internal defects and performing repairs. Comprehensive testing usually requires a plethora of crafted and representative test cases, in which they can evaluate whether the trained DL-based systems can generalize to the unseen data. Many different test input generators (TIGs) have been proposed, creating a massive amount of inputs. Hence, test selection metrics are proposed for selecting and prioritizing potentially mispredicted inputs, thereby saving manual labelling costs and accelerating Deep Neural Networks (DNNs) repairing process. Although existing metrics have shown promising results, we have observed three limitations when applying them in practice: 1. Lack the consideration of consensus decision from different metrics; 2. The evaluation criteria are insufficient; 3. They mainly focus on simple image classification tasks (e.g., MNIST, SVHN). In this paper, we address these three limitations correspondingly by performing ensemble-based testing through granularity and scale unification, examining the effectiveness of test selection metrics on both the capability of root cause detection and the capability
of retraining guidance, and assessing the performance on two different domains, autonomous driving and malware detection. Validated on two Android malware datasets and two real-world driving datasets, our method outperforms five baselines across two criteria. Compared with the best baseline, for retraining guidance, our method reaches an extra accuracy improvement by up to 9.13\% for driving datasets and 2.81\% for malware datasets. For root cause detection, our method is estimated to identify more root causes in 23 out of 44 cases across all datasets. Additionally, our ablation study validates the effectiveness of each component of our method, including scale unification and ensemble-based selection. We also open-source our code to assist further research for the Software Engineering community.


#Environment:
To successfully run the code, we provide two conda environment files. 
-environmentdnn.yml is the file for the main experiments except for computing the root causes.
-environmentdetect.yml is the file for conducting clustering and computing the number of root causes with the given test suite.

Start by cloning the repository on your local folder and then, to install, just run:
`conda env create -f environmentdnn.yml` 
`conda env create -f environmentdetect.yml` 

#Dataset:
##Driving data (Dave, Udacity):
For Dave, please follow the link to download:
https://github.com/SullyChen/Autopilot-TensorFlow

For Udacity, please follow the official procedure to download (Ch2_002): 
https://github.com/udacity/self-driving-car/tree/master/datasets

##Android Packages (AndroZoo, Drebin):
For AndroZoo, please follow the procedure from the official website to download:
https://androzoo.uni.lu/

For Drebin, we directly use the extracted feature, provided by the authors of the paper: 
Adversarial Deep Ensemble: Evasion Attacks and Defenses for Malware Detection (TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY), by Deqiang Li and Qianmu Li.

#Run:
##Test selection metrics to guide retraining:
`conda activate dnntest`

For malware data, 
`cd adv-dnn-ens-malware`
`python retrain_selected.py` for adversarial OOD operators.
`python retrain_selected_outofsrc.py` for out-of-src OOD operators.

For driving data, 
`cd adv-self-driving`
`python retrain_selected.py` for adversarial OOD operators.
`python retrain_selected_outofsrc.py` for out-of-src OOD operators.

Please specify the interested parameters (metric, budget, id_ratio, OOD method, epochs) in the script.

##Test selection metrics to detect root causes:
`conda activate faultdetect`

For malware data, 
`cd adv-dnn-ens-malware`
`python3 root_cause_detect_malware_gpu.py`

For driving data, 
`cd adv-self-driving`
`python3 root_cause_detect_gpu.py`



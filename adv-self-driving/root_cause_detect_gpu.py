'''
usage python3 root_cause_detect_gpu.py
we use combined selected test suite from all operators to examine the fault detection
'''

from dataloader.data_utils import *
# Plotting keywords
plot_kwds = {'alpha': 0.15, 's': 80, 'linewidths': 0}
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from DBCV.DBCV import DBCV
from scipy.spatial.distance import euclidean
def plot_clusters(data, algorithm, args, kwds, idr):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    if len(np.unique(labels)) == 1:
        return labels
    label_copy = copy.deepcopy(labels)
    label_list = list(label_copy)
    sorted_labels = sorted(label_copy)

    for i in range(-1,  np.array(sorted_labels).max() + 1):
        count = label_list.count(i)

    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds) # plot the first two dimensions
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title(f'Clusters found by {algorithm.__name__}', fontsize=24)
    plt.text(-0.5, 0.7, f'Clustering took {end_time - start_time:.2f} s', fontsize=14)
    plt.savefig(f'cluster_figs/cluster_plot_{metric}_bg{bg}_idr{idr}.png')
    plt.clf()
    return labels

# min-max rescale the last two features:
def scale_one(X):
    if len(np.unique(X)) == 1:
        if np.unique(X) != 0:
            return X / X.max()
        else:
            return X
    else:
        return (X - X.min()) / (X.max() - X.min())
# Dimensionality Reduction
import hdbscan
import sklearn.metrics
# VGG feature extraction
from vgg import *
import torchvision
import torch
import pandas as pd
from cuml.manifold import UMAP
from sklearn.preprocessing import StandardScaler
def umap_gpu(ip_mat, min_dist, n_components, n_neighbors, metric):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    scaler = StandardScaler()
    ip_std = scaler.fit_transform(ip_mat)
    reducer = UMAP(min_dist=min_dist, n_components=n_components, n_neighbors=n_neighbors, metric=metric)
    umap_embed = reducer.fit_transform(ip_std)

    return umap_embed

transform = torchvision.transforms.ToTensor()
extractor = vgg16(pre_trained=True, require_grad=False)

def root_cause(dataset, model_name, bg, metric, weighted, weights):
    best_results = []
    for idr in id_ratio:
        test_suite, gt_lbls, pred_lbls = [], [], []
        for operator in operators:
            if weighted==True:
                path = f'/adv-self-driving/test_suite/{dataset}/{model_name}/{operator}/{metric}/{weights}/{bg}/{idr}/data.txt'
            else:
                path = f'/adv-self-driving/repred_test_suite/{dataset}/{model_name}/{operator}/{metric}/{bg}/{idr}/data.txt'
            with open(path, 'r') as file:
                lines = file.readlines()
                max_samples = 4000 * bg
                samples = min(len(lines), int(max_samples))
                for line in lines[:samples]:
                    test_suite.append(line.split(' ')[0])
                    gt_lbls.append(int(line.split(' ')[1]))
                    pred_lbls.append(int(line.split(' ')[2].split('\n')[0]))
        X = [np.asarray(Image.open(os.path.join(x))) for x in test_suite]
        # get the misprediction set
        mis_idx = np.array(gt_lbls) != np.array(pred_lbls)
        X = np.array(X)[mis_idx]
        gt_lbls = np.array(gt_lbls)[mis_idx]
        pred_lbls = np.array(pred_lbls)[mis_idx]
        mispred = int(sum(mis_idx))
        bb, trace, hdbscan_in_umap, clustering_results = [], [], [], []
        if mispred <= 2:
            best_results.append(pd.DataFrame({"Number of Clusters": 0,
                                              "Silhouette Score": -1.0,
                                              "Combined Score": -1.0,
                                              "Number of Noisy Inputs": mispred,
                                              "Config": [-1,-1,-1,-1,-1.0],
                                              "id ratio": idr,
                                              "dataset": dataset,
                                              "metric": metric,
                                              "bg": bg,
                                              "model": model}).iloc[0])
            continue

        suite_feat = []
        for sample, gt_label, pred_label in zip(X, gt_lbls, pred_lbls):
            feat = extractor(transform(sample))
            last_feat = torch.mean(feat[-1], (1,2))
            concat_feat = np.concatenate((last_feat, [gt_label], [pred_label]))
            suite_feat.append(concat_feat)
        suite_feat = np.vstack(suite_feat)
        PY_scaled = scale_one(suite_feat[:, -1])
        suite_feat[:, -1] = PY_scaled
        TY_scaled = scale_one(suite_feat[:, -2])
        suite_feat[:, -2] = TY_scaled
        X_features = suite_feat[:, :-2]
        Sumn = 0

        if bg == 0.05:
            i_list, j_list = [250, 200, 150, 100], [200, 150, 100, 50]
        elif bg == 0.1:
            i_list, j_list = [400, 350, 300, 250, 200], [350, 300, 250, 200, 150]
        elif bg == 0.03:
            i_list, j_list = [150, 100, 60], [100, 50, 30]
        elif bg == 0.01:
            i_list, j_list = [60], [30]

        for i, j in zip(i_list, j_list):
            if i >= suite_feat.shape[0]:
                i = suite_feat.shape[0] - 5
                j = i - 10
                if j <= 0 or i <= 0:

                    i = suite_feat.shape[0] - 1
                    j = i - 1
            for k, o in zip([5, 10, 20, 25], [3, 5, 15, 20]):
                for n_n in [0.03, 0.05, 0.1, 0.2, 0.25, 0.3]:

                    # UMAP Dimensionality Reduction
                    u1 = umap_gpu(ip_mat=suite_feat, min_dist=n_n, n_components=i, n_neighbors=k, metric='Euclidean')
                    u = umap_gpu(ip_mat=u1, min_dist=0.1, n_components=j, n_neighbors=o, metric='Euclidean')
                    u = np.c_[u, TY_scaled, PY_scaled]


                    labels = plot_clusters(u, hdbscan.HDBSCAN, (), {'min_cluster_size': 5}, idr)
                    if len(np.unique(labels))==1:
                        config = [i, j, k, o, n_n]
                        clustering_results.append({
                            "Number of Clusters": labels.max() + 1,
                            "Silhouette Score": -1,
                            "DBCV Score": -1,
                            "Combined Score": -1,  # 0.5 * silhouette_umap + 0.5 * DBCV_score,
                            "Number of Noisy Inputs": list(labels).count(-1),
                            "Config": config,
                            "id ratio": idr,
                            "dataset": dataset,
                            "metric": metric,
                            "bg": bg,
                            "model": model,
                        })
                        continue
                    # evaluate the clustering
                    silhouette_umap = sklearn.metrics.silhouette_score(u, labels)
                    silhouette_features = sklearn.metrics.silhouette_score(X_features, labels)
                    DBCV_score = DBCV(u, labels, dist_function=euclidean)

                    if (silhouette_umap >= 0.1 or silhouette_features >= 0.1):
                        config = [i, j, k, o, n_n]

                        Sumn += 1

                        clustering_results.append({
                            "Number of Clusters": labels.max() + 1,
                            "Silhouette Score": silhouette_umap,
                            "DBCV Score": DBCV_score,
                            "Combined Score": 0.5 * silhouette_umap + 0.5 * DBCV_score,
                            "Number of Noisy Inputs": list(labels).count(-1),
                            "Config": config,
                            "id ratio": idr,
                            "dataset": dataset,
                            "metric": metric,
                            "bg": bg,
                            "model": model,
                        })
        # Display clustering results in a table and select the one config clustering that has best Silhouette/DBCV score
        clustering_df = pd.DataFrame(clustering_results)
        # Select the best clustering based on the combined score
        idx_clustering = np.argmax(clustering_df['Combined Score'])
        best_results.append(clustering_df.iloc[idx_clustering])

    if weighted==True:
        root_path = f'root_cause/{dataset}/{model_name}/{metric}/{weights}/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        pd.DataFrame(best_results).to_csv(root_path+f'bg{bg}.csv')
    else:
        root_path = f'repred_root_cause/{dataset}/{model_name}/{metric}/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        pd.DataFrame(best_results).to_csv(root_path + f'bg{bg}.csv')
if __name__ == '__main__':
    params = {
            'dataset': ['Udacity'], #DAVE, Udacity
            'model_name': ['Dave2V1', 'Dave2V2'],
            'selection_metric': ['deepgini'], #['random', 'gd', 'entropy', 'deepgini', 'dat_ood_detector'],#, 'random', 'gd', 'entropy', 'deepgini'],
            'budget': [0.01, 0.03, 0.05, 0.1],
        }
    id_ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    operators = [
        'BIM',
        'FGSM',
        'Mementum',
        'PGD',
        'out-of-src']

    for dataset in params['dataset']:
        for model in params['model_name']:
            for bg in params['budget']:
                for metric in params['selection_metric']:
                    check_path = f'repred_root_cause/{dataset}/{model}/{metric}/' + f'bg{bg}.csv'
                    root_cause(dataset, model, bg, metric, False, '0.5_0.5_60')

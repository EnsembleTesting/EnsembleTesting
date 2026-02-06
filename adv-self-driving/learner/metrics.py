import sys
sys.path.append('/adv-self-driving')

import numpy as np
from dataloader import data_utils
import tensorflow.compat.v1 as tf
import scipy.stats as stats

BATCH_SIZE = 1
def compute_gd(sess, candidateX, candidatey, model):

    min_max_scaler = preprocessing.MinMaxScaler()
    # test_input = utils.DataProducer(candidateX, candidatey, batch_size=1, name='test')
    test_input = data_utils.val_generator(candidateX, candidatey, batch_size=1)

    if model.model_name.lower() == 'dave2v1':
        hidden_size = 582
    elif model.model_name.lower() == 'dave2v2':
        hidden_size = 582
    elif model.model_name.lower() == 'dave2v9':
        hidden_size = 500

    with sess.as_default():
        feat = np.zeros((1, hidden_size))
        for [x, y] in test_input:  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            _feat = sess.run(model.dense1, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        # normalize group
        normalize_select_group = min_max_scaler.fit_transform(feat_mat)
        # compute GD
        GD = np.linalg.det(np.matmul(normalize_select_group, normalize_select_group.T))
    return GD

def to_ordinal(y):
    """
    Convert labels to 1D int array.
    Works for:
      - scalar labels
      - one-hot labels
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int32)
    # one-hot or probability
    return np.argmax(y, axis=1).astype(np.int32)

def _pairwise_cdist(x1, x2):
    """
    Fallback if scipy not available.
    x1: (1, D)
    x2: (N, D)
    return: (N,)
    """
    # ||a-b||2 = ||a||2 + ||b||2 - 2 a.b
    a2 = np.sum(x1 * x1, axis=1, keepdims=True)       # (1,1)
    b2 = np.sum(x2 * x2, axis=1, keepdims=True).T     # (1,N)
    ab = np.dot(x1, x2.T)                              # (1,N)
    d2 = a2 + b2 - 2.0 * ab
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).ravel()

class DSA(object):
    def __init__(self, sess, trainX, trainy, model, dataset_name, model_name,
                 batch_size=128, eps=1e-5):
        """
        sess: TF1 session
        train: training inputs
        label: training labels (scalar or one-hot)
        model: your TF1 model wrapper (must provide x_input, y_input, is_training)
        layers: list of (name, tensor)
        dataset_name/model_name: used for cache file name
        """
        self.sess = sess
        self.trainX, self.trainy = trainX, trainy
        self.model = model
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.eps = eps

        # Use last hidden layer (NOT softmax) => layers[-2]
        # Each element is (layer_name, layer_tensor)
        self.feat_layer_tensor = self.model.dense2

        # label to ordinal
        self.train_label = to_ordinal(trainy)
        trainset_name = dataset_name

        if not os.path.exists('./AllResult/DSA/'):
            os.makedirs('./AllResult/DSA/')

        self.train_act_path = './AllResult/DSA/{}_{}_neuron_activate_train.npy'.format(
            trainset_name, model_name
        )

        # load or compute training activations
        if os.path.exists(self.train_act_path):
            self.neuron_activate_train = np.load(self.train_act_path)
        else:
            self.neuron_activate_train = self._compute_activations(self.trainX, self.trainy, self.feat_layer_tensor)
            np.save(self.train_act_path, self.neuron_activate_train)

    def _postprocess_activation(self, act):
        """
        Convert activation to (B, C) feature vectors.
        - conv: (B,H,W,C) -> mean over H,W
        - seq:  (B,T,C)   -> mean over T
        - dense:(B,C)     -> keep
        """
        act = np.asarray(act)
        if act.ndim == 4:
            # (B,H,W,C)
            return np.mean(act, axis=(1,2))
        elif act.ndim == 3:
            # (B,T,C)
            return np.mean(act, axis=1)
        elif act.ndim == 2:
            return act
        else:
            # unexpected, flatten
            return act.reshape((act.shape[0], -1))

    def _compute_activations(self, X, y, layer_tensor):
        test_input = data_utils.val_generator(X, y, batch_size=128)
        feats = []
        with self.sess.as_default():
            for [x, y] in test_input:
                feed = {
                    self.model.x_input: x,
                    self.model.y_input: y,
                    self.model.is_training: False
                }
                act = self.sess.run(layer_tensor, feed_dict=feed)
                act = self._postprocess_activation(act)
                feats.append(act)
        return np.concatenate(feats, axis=0)

    def fit(self, test, label):
        """
        Return DSA scores for test set.
        DSA(x) = dist_a / (dist_b + eps)
        dist_a: min distance to training sample with SAME label
        dist_b: min distance to training sample with DIFFERENT label
        """
        label = to_ordinal(label)

        # cache score file
        save_path = './AllResult/DSA/{}_{}_dsa.npy'.format(self.dataset_name, self.model_name)

        if os.path.exists(save_path):
            return np.load(save_path)

        # compute test activations
        neuron_activate_test = self._compute_activations(test, label, self.feat_layer_tensor)

        all_train_acts = self.neuron_activate_train
        all_train_labels = self.train_label

        test_score = []
        for i in range(len(neuron_activate_test)):
            x = neuron_activate_test[i].reshape(1, -1)
            y = label[i]

            # distance to all training acts
            if _HAS_SCIPY:
                all_dists = cdist(x, all_train_acts).ravel()
            else:
                all_dists = _pairwise_cdist(x, all_train_acts)

            same_mask = (all_train_labels == y)
            diff_mask = (all_train_labels != y)

            # guard in case of weird label distribution
            if np.any(same_mask):
                dist_a = np.min(all_dists[same_mask])
            else:
                dist_a = np.min(all_dists)
            if np.any(diff_mask):
                dist_b = np.min(all_dists[diff_mask])
            else:
                dist_b = np.max(all_dists)  # fallback
            test_score.append(dist_a / (dist_b + self.eps))
        test_score = np.asarray(test_score, dtype=np.float32)
        np.save(save_path, test_score)
        return test_score


def dsa_select(sess, trainX, trainy, X, y, model, budget, dataset, model_name,
              batch_size=128):
    """
    Select top samples by DSA score (descending).
    budget: ratio (0~1), consistent with your deepgini code.
    """
    dsa_model = DSA(sess=sess, trainX=trainX, trainy=trainy, model=model, dataset_name=dataset, model_name=model_name,
                    batch_size=batch_size)

    scores = dsa_model.fit(X, y)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_X = np.asarray(X)[selected_idx]
    selected_y = np.asarray(y)[selected_idx]
    return selected_X, selected_y
                  
def deepgini(sess, candidateX, candidatey, model, budget):
    #test_input = utils.DataProducer(candidateX, candidatey, batch_size=120, name='test')
    test_input = data_utils.val_generator(candidateX, candidatey, batch_size=BATCH_SIZE)
    gini = 1 - tf.reduce_sum(model.y_proba ** 2, axis=1)

    with sess.as_default():
        score = np.array([0.0])
        for [x, y] in test_input:
            if 128 not in x.shape:
                print(x, y)
                print(x.shape, y.shape)
                import pdb; pdb.set_trace()
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }

            _gini_score = sess.run(gini, feed_dict=test_dict)
            score = np.hstack((score, _gini_score.squeeze()))

            #score.append(_gini_score.squeeze())

    scores = np.array(score[1:]).flatten()
    # select ratio from the largest scores
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = np.array(candidateX)[selected_idx]
    selected_candidatey = np.array(candidatey)[selected_idx]
    return selected_candidateX, selected_candidatey, scores  # Gini scores for all test inputs


def entropy(sess, candidateX, candidatey, model, budget):
    test_input = data_utils.val_generator(candidateX, candidatey, batch_size=BATCH_SIZE)
    entropies = -1 * tf.reduce_sum(model.y_proba * tf.log(tf.clip_by_value(model.y_proba, 1e-8, 1.0)), axis=1) # we have clipped the prob to min 1e-8 to avoid Nan when prob=0
    with sess.as_default():
        score = np.array([0.0])
        for [x, y] in test_input:
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }

            _ent_score = sess.run(entropies, feed_dict=test_dict)
            score = np.hstack((score, _ent_score.squeeze()))

    scores = np.array(score[1:]).flatten()
    # select ratio from the largest scores
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = np.array(candidateX)[selected_idx]
    selected_candidatey = np.array(candidatey)[selected_idx]
    return selected_candidateX, selected_candidatey, scores # entropy scores for all test inputs

def Random(sess, candidateX, candidatey, budget):
    selection_size = int(budget * candidateX.shape[0])
    select_idx = np.random.choice(np.arange(len(candidateX)), selection_size)
    selected_candidateX, selected_candidatey = candidateX[select_idx], candidatey[select_idx]

    return selected_candidateX, selected_candidatey

from sklearn import preprocessing
def GD(sess, candidateX, candidatey, model, selection_size, number=60):
    '''
    Black-box diversity test selection metric

    Step 1: Feature Extraction. Extract features for each sample in the candidateX -> Shape (len(canX), m)
    Step 2: Randomly select with replacement N=number groups of size n=budget*len(idx) from the feature matrix.
    Step 3: For each group -> Nomralize column-wise (feature-wise)
    Step 4: Calculate diversity score GD (geometric diversity)
    Step 5: Select the group with the highest GD.
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    test_input = data_utils.val_generator(candidateX, candidatey, batch_size=BATCH_SIZE)
    #selection_size = int(budget * candidateX.shape[0])

    if model.model_name.lower() == 'dave2v1':
        hidden_size = 582
    elif model.model_name.lower() == 'dave2v2':
        hidden_size = 582

    with sess.as_default():
        feat = np.zeros((1,hidden_size))
        for [x,y] in test_input:  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }

            _feat = sess.run(model.dense1, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        GD_scores, selected_indices = [], []
        for _ in range(number):
            select_idx = np.random.choice(np.arange(len(feat_mat)), selection_size)
            selected_indices.append(select_idx)
            select_group = feat_mat[select_idx] #shape (bg*len, feat)
            # normalize group
            normalize_select_group = min_max_scaler.fit_transform(select_group)
            # compute GD
            GD = np.linalg.det(np.matmul(normalize_select_group, normalize_select_group.T))
            GD_scores.append(GD.squeeze())

        max_idx = np.argmax(np.array(GD_scores))
        chosen_indices = selected_indices[max_idx]
        selected_candidateX, selected_candidatey = candidateX[chosen_indices], candidatey[chosen_indices]

    return selected_candidateX, selected_candidatey

def dat_ood_detector(sess, trainX, trainy, candidateX, candidatey, hybrid_test, hybrid_testy, model, budget):
    '''
   introduce OOD detector: using the first hidden-layer feature
   '''
    if model.model_name.lower() == 'dave2v1':
        hidden_size = 582
    elif model.model_name.lower() == 'dave2v2':
        hidden_size = 582
    elif model.model_name.lower() == 'dave2v9':
        hidden_size = 500

    train_input = data_utils.val_generator(trainX, trainy, batch_size=BATCH_SIZE, mode='origin')
    can_input = data_utils.val_generator(candidateX, candidatey, batch_size=BATCH_SIZE, mode='rescaled')

    with sess.as_default():
        feat = np.zeros((1, hidden_size))
        feat_can = np.zeros((1, hidden_size))
        candidate_prediction_label = np.array([0.0])
        for [x, y] in train_input:  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense1, feed_dict=test_dict)

            # _feat = _feat.reshape((BATCH_SIZE,-1))

            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        print('print train input too many?')
        for [x, y] in can_input:  # not shuffle
            can_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense1, feed_dict=can_dict)
            lbl = sess.run(model.y_pred, feed_dict=can_dict)

            candidate_prediction_label = np.hstack((candidate_prediction_label, lbl))

            feat_can = np.vstack((feat_can, _feat))

        feat_mat_can = feat_can[1:]
        candidate_prediction_label = candidate_prediction_label[1:]

    # Use train set to compute center of the hypersphere, it is defined as the mean feature of training data (ID data)
    center = np.mean(feat_mat, axis=0)
    # compute distance of each train feature to center
    dist = [np.sum((i - center) ** 2) for i in feat_mat]
    dist_sort = np.sort(dist)  # from smallest to biggest
    threshold = dist_sort[int(0.95 * (len(dist_sort)))]  # select threshold such that 95% train data are correctly classified as ID
    print('print threshold')
    dist_can = [np.sum((i - center) ** 2) for i in feat_mat_can]
    candidateX_id, candidateX_ood = candidateX[dist_can <= threshold], candidateX[dist_can > threshold]
    candidatey_id, candidatey_ood = candidatey[dist_can <= threshold], candidatey[dist_can > threshold]

    select_size = budget * candidateX.shape[0]
    id_select_num = int(candidateX_id.shape[0] * budget)  # int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(candidateX_ood.shape[0] * budget)  # int(select_size - id_select_num)
    para_there = 0.5  # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX_ood.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size:  # this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size:  # this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)

    # select ID by DeepGini
    _, _, id_scores = deepgini(sess, candidateX_id, candidatey_id, model, budget)
    # select ratio from the largest scores
    idx = np.argsort(id_scores)[::-1]
    id_selected_idx = idx[:int(id_select_num)]
    selected_candidateX_id = candidateX_id[id_selected_idx]
    selected_candidatey_id = candidatey_id[id_selected_idx]

    # ood data selection
    # top 1 label
    # candidate_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: candidateX, model.y_input: candidatey, model.is_training: False})  #np.argmax(model.predict(candidateX), axis=1)
    hybrid_input = data_utils.val_generator(hybrid_test, hybrid_testy, batch_size=BATCH_SIZE, mode='rescaled')
    with sess.as_default():
        reference_prediction_label = np.array([0.0])
        for [x, y] in hybrid_input:  # not shuffle
            hybrid_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            lbl = sess.run(model.y_pred, feed_dict=hybrid_dict)

            reference_prediction_label = np.hstack((reference_prediction_label, lbl))
    reference_prediction_label = reference_prediction_label[1:]
    # reference_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: hybrid_test, model.y_input: hybrid_testy, model.is_training: False}) #np.argmax(model.predict(hybrid_test), axis=1)
    # print(reference_prediction_label)
    reference_labels = []
    for i in range(0, 3):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidateX) / select_size
    reference_labels = reference_labels / s_ratio

    ood_part_index = np.where((dist_can > threshold) == True)[0]

    label_list = []
    index_list = []
    print('print 4')
    if ood_select_num != 0:
        for _ in range(1000):
            ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)
            this_labels = candidate_prediction_label[ood_select_index.astype(np.int)]
            single_labels = []
            for i in range(0, 3):
                label_num = len(np.where(this_labels == i)[0])
                single_labels.append(label_num)
            index_list.append(ood_select_index)
            label_list.append(single_labels)
        index_list = np.asarray(index_list)
        label_list = np.asarray(label_list)
        print('print 5')
        # compare to test label
        label_minus = np.abs(label_list - reference_labels)
        var_list = np.sum(label_minus, axis=1)
        var_list_order = np.argsort(var_list)

        ood_select_index = index_list[var_list_order[0]]
        selected_candidateX_ood = candidateX[ood_select_index]
        selected_candidatey_ood = candidatey[ood_select_index]
        selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
        selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)
    else:
        selected_data = selected_candidateX_id
        selected_label = selected_candidatey_id
    print('print 6')
    return selected_data, selected_label  # selected candidate data for retraining

def _tail_p_value_normal(x, mu, sigma, tail):
    """
    Tail-aware p-value under Normal(mu, sigma).
    tail: 'upper' => P(X >= x)
          'lower' => P(X <= x)
    """
    # avoid sigma=0 causing NaNs
    if sigma <= 1e-12:
        # if distribution is degenerate, treat everything as not-extreme
        # (or you can return 1.0 for all)
        return np.ones_like(x, dtype=np.float64)

    z_cdf = stats.norm.cdf(x, mu, sigma)
    if tail == 'lower':
        p = z_cdf
    else:
        p = 1.0 - z_cdf
    return p

def _fisher_combine_pvals(pvals_list, eps=1e-300):
    """
    Fisher's method combining multiple p-values.
    pvals_list: list of arrays with same shape, e.g. [p1, p2, ...]
    Returns:
      p_fisher: array of same shape
      chi2_stat: array of same shape
    """
    # clip to avoid log(0)
    log_terms = 0.0
    for p in pvals_list:
        p_clip = np.clip(p.astype(np.float64), eps, 1.0)
        log_terms += np.log(p_clip)

    chi2_stat = -2.0 * log_terms
    dof = 2 * len(pvals_list)
    # survival function gives P(Chi2 >= stat)
    p_fisher = stats.chi2.sf(chi2_stat, dof)
    return p_fisher, chi2_stat
def ensemble_p_values_fisher(sess, trainX, trainy, candidateX, candidatey, model, budget, our_method_params):
    '''
    Use p-values + Fisher combination (tail-aware).
    Does not utilize OOD detector to distinguish ID & OOD.
    Uses Fisher-combined p-values to form a risk-biased pool, then GD for diversity.
    '''
    num = our_method_params['gd_num']
    ratio = our_method_params['ratio']
    selection_size = int(budget * candidateX.shape[0])

    # --- NEW: tail direction for each metric (default to upper tail) ---
    # 'upper': larger score => more extreme; 'lower': smaller score => more extreme
    tail_gini = our_method_params.get('tail_gini', 'upper')
    tail_ent  = our_method_params.get('tail_ent',  'upper')

    # compute scores
    _, _, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    _, _, train_scores_gini = deepgini(sess, trainX[:5000], trainy[:5000], model, budget)

    _, _, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)
    _, _, train_scores_entropy = entropy(sess, trainX[:5000], trainy[:5000], model, budget)

    # fit reference distribution (your current assumption: Normal with empirical mean/std)
    train_scores_gini = train_scores_gini.squeeze()
    train_scores_entropy = train_scores_entropy.squeeze()

    mu_gini, sigma_gini = stats.tmean(train_scores_gini), stats.tstd(train_scores_gini)
    mu_ent,  sigma_ent  = stats.tmean(train_scores_entropy), stats.tstd(train_scores_entropy)

    # tail-aware p-values
    p_values_gini = _tail_p_value_normal(id_scores_gini, mu_gini, sigma_gini, tail_gini)
    p_values_ent  = _tail_p_value_normal(id_scores_entropy, mu_ent, sigma_ent, tail_ent)

    # Fisher combine (smaller => more extreme across both metrics)
    p_fisher, chi2_stat = _fisher_combine_pvals([p_values_gini, p_values_ent])

    # rank by Fisher combined p-value
    idx = np.argsort(p_fisher)  # from smallest (most extreme) to largest
    half_num = int(ratio * len(idx))

    # GD: select most diverse set from the risk-biased pool
    if half_num > selection_size:
        poolX = candidateX[idx[:half_num]]
        pooly = candidatey[idx[:half_num]]
        selected_candidateX, selected_candidatey = GD(sess, poolX, pooly, model, selection_size, number=num)
    else:
        selected_candidateX, selected_candidatey = GD(sess, candidateX, candidatey, model, selection_size, number=num)
    return selected_candidateX, selected_candidatey

def deepGD(sess, candidateX, candidatey, model, selection_size):
    import random
    cxpb = 0.9
    size_ind = selection_size
    ngen = 20
    pop_size = 70
    pop_size = (pop_size // 4) * 4
    tournsize=2
    mutpb = 0.7
    def get_feature_prob(sess, candidateX, candidatey, model):
        # min_max_scaler = preprocessing.MinMaxScaler()
        test_input = utils.DataProducer(candidateX, candidatey, batch_size=120, name='test')
        # selection_size = int(budget * candidateX.shape[0])
        gini = 1 - tf.reduce_sum(model.y_proba ** 2, axis=1)
        if model.model_name.lower() == 'deepdrebin':
            hidden_size = 200
        elif model.model_name.lower() == 'basic_dnn':
            hidden_size = 160

        with sess.as_default():
            score = np.array([0.0])
            feat = np.zeros((1, hidden_size))
            # prob = np.zeros((1, 2))
            for _, x, y in test_input.next_batch():  # not shuffle
                test_dict = {
                    model.x_input: x,
                    model.y_input: y,
                    model.is_training: False
                }
                # Output_probability = sess.run(model.y_proba, feed_dict=test_dict)
                _feat = sess.run(model.dense2, feed_dict=test_dict)
                feat  = np.vstack((feat, _feat))
                _gini_score = sess.run(gini, feed_dict=test_dict)
                score = np.hstack((score, _gini_score.squeeze()))
                # prob = np.vstack((prob, Output_probability))
            feat_mat = feat[1:]
            # prob_mat = Output_probability[1:]
        scores = np.array(score[1:]).flatten()
        return feat_mat, scores

    allowed = list(range(len(candidateX)))  # fallback: allow all
    features, Gini_scores = get_feature_prob(sess, candidateX, candidatey, model)  # <-- provide below or reuse yours

    def gd_logdet(ids):
        sel = features[np.array(ids, dtype=int)]
        gram = np.dot(sel, sel.T)
        sign, logdet = np.linalg.slogdet(gram + 1e-12 * np.eye(gram.shape[0]))
        # if numerical issues, penalize
        if sign <= 0:
            return -1e9
        return float(logdet)

    def evaluate(ind):
        # avg gini
        s = 0.0
        for idx in ind:
            s += float(Gini_scores[int(idx)])
        avg_g = s / float(len(ind))

        div = gd_logdet(ind)
        return avg_g, div
    # -----------------------------
    # DEAP NSGA-II setup (Py2)
    # -----------------------------
    from deap import base, creator, tools

    # create types only once (safe guard if rerun in same process)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    def init_individual():
        return creator.Individual(random.sample(allowed, size_ind))

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Crossover: partial mix, then repair to keep unique and length=size_ind
    def cx_repair(ind1, ind2):
        a = list(ind1)
        b = list(ind2)
        point = random.randint(1, size_ind - 1)

        child1 = a[:point] + b[point:]
        child2 = b[:point] + a[point:]

        def repair(child):
            # keep order, drop dups
            seen = set()
            out = []
            for x in child:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            # fill missing with random allowed not in out
            if len(out) < size_ind:
                pool = list(set(allowed) - set(out))
                out.extend(random.sample(pool, size_ind - len(out)))
            return creator.Individual(out[:size_ind])

        return repair(child1), repair(child2)

    # Mutation: replace a few low-gini genes with random new indices
    def mut_replace(ind, n_least=5):
        # sort indices in ind by gini ascending; replace among the lowest n_least
        ranked = sorted(ind, key=lambda i: float(Gini_scores[int(i)]))
        to_replace = ranked[:n_least]
        pool = list(set(allowed) - set(ind))
        if not pool:
            return (ind,)
        for old in to_replace:
            if pool:
                new = random.choice(pool)
                ind[ind.index(old)] = new
                pool.remove(new)
        # ensure uniqueness (repair if needed)
        if len(set(ind)) < size_ind:
            ind[:] = list(dict.fromkeys(ind))  # preserve order, drop dups
            pool2 = list(set(allowed) - set(ind))
            if len(ind) < size_ind:
                ind.extend(random.sample(pool2, size_ind - len(ind)))
        return (ind,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_repair)
    toolbox.register("mutate", mut_replace)
    toolbox.register("select", tools.selNSGA2)

    # Initialize
    pop = toolbox.population(n=pop_size)
    # Evaluate initial
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))
    print('start evolution')
    # Evolution
    for gen in range(ngen):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover/mutation
        for i in range(0, len(offspring), 2):
            if random.random() < cxpb and i + 1 < len(offspring):
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1])

        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])

        # Re-evaluate
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # NSGA-II environmental selection
        pop = toolbox.select(pop + offspring, pop_size)
    print('finish evolution')
    # Pareto front
    front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    print('finish pareto front')
    # Pick one solution from Pareto front (simple rule: highest avg_gini, tie-break by div)
    best = max(front, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1]))
    selected_candidateX = candidateX[best]
    selected_candidatey = candidatey[best]

    return selected_candidateX, selected_candidatey  # Gini scores for all test inputs

def ensemble_simple(sess, candidateX, candidatey, model, budget, our_method_params):
    '''
    No p-values.
    Does not utilize OOD detector to distinguish ID & OOD
    just use GD to select from uncertain batch by Entropy
    '''
    num, ratio = our_method_params['gd_num'], our_method_params['ratio']
    selection_size = int(budget * candidateX.shape[0])

    _,_, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)

    idx = np.argsort(id_scores_gini)[::-1]
    half_num = int(ratio * len(idx))
    # GD: select most diverse set from the uncertain pool
    if half_num > selection_size:
        selected_candidateX, selected_candidatey = GD(sess, candidateX[idx[:half_num]], candidatey[idx[:half_num]], model, selection_size, number=num)
    else:
        selected_candidateX, selected_candidatey = GD(sess, candidateX,
                                                            candidatey, model, selection_size,
                                                            number=num)

    return selected_candidateX, selected_candidatey # selected candidate data for retraining

def ensemble_simple_add(sess, candidateX, candidatey, model, budget, our_method_params):
    '''
    No p-values.
    Does not utilize OOD detector to distinguish ID & OOD
    just use GD to select from uncertain batch by Entropy
    '''
    num, ratio = our_method_params['gd_num'], our_method_params['ratio']
    selection_size = int(budget * candidateX.shape[0])

    _,_, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)

    idx = np.argsort(id_scores_gini+id_scores_entropy)[::-1]
    half_num = int(ratio * len(idx))
    # GD: select most diverse set from the uncertain pool
    if half_num > selection_size:
        selected_candidateX, selected_candidatey = GD(sess, candidateX[idx[:half_num]], candidatey[idx[:half_num]], model, selection_size, number=num)
    else:
        selected_candidateX, selected_candidatey = GD(sess, candidateX,
                                                            candidatey, model, selection_size,
                                                            number=num)

    return selected_candidateX, selected_candidatey # selected candidate data for retraining


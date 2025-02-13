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

def ensemble_p_values(sess, trainX, trainy, candidateX, candidatey, model, budget, our_method_params):
    '''
    Use p-values.
    Does not utilize OOD detector to distinguish ID & OOD
    just use GD to select from uncertain batch (by p-values)
    '''
    w_gini, w_ent, num, ratio = our_method_params['w_gini'], our_method_params['w_ent'], our_method_params['gd_num'], our_method_params['ratio']
    selection_size = int(budget * candidateX.shape[0])

    _,_, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    _,_, train_scores_gini = deepgini(sess, trainX[:5000], trainy[:5000], model, budget)
    _, _, train_scores_entropy = entropy(sess, trainX[:5000], trainy[:5000], model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)

    # select ratio from the largest scores
    mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    weighted_p_vals = w_gini * p_values_gini + w_ent * p_values_entropy
    idx = np.argsort(weighted_p_vals)  # from smallest (most uncertain) to biggest
    half_num = int(ratio * len(idx))
    # GD: select most diverse set from the uncertain pool

    if half_num > selection_size:
        selected_candidateX, selected_candidatey = GD(sess, candidateX[idx[:half_num]], candidatey[idx[:half_num]], model, selection_size, number=num)
    else:
        selected_candidateX, selected_candidatey = GD(sess, candidateX,
                                                            candidatey, model, selection_size,
                                                            number=num)

    return selected_candidateX, selected_candidatey # selected candidate data for retraining

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

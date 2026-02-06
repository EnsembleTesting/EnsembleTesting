import numpy as np
import pandas as pd
import os
import csv
from config import config
from learner import model_scope_dict
from dataloader.data_utils import LabeledDaveDataset,\
    DaveDataset, load_train_data

our_method_params = {}
our_method_params['gd_num'], our_method_params['w_gini'], our_method_params['w_ent'], our_method_params['ratio'] = 60, 0.0, 1.0, 0.5

tests = [

{
        'selection_metric': ['ensemble_p_values_fisher'],#['entropy', 'random', 'gd', 'deepgini'],
        'budget':  [0.1, 0.05, 0.03, 0.01],
        'id_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'method': ['FGSM', 'BIM', 'Mementum', 'PGD'], # 'FGSM', 'BIM', 'Mementum', 'PGD'
        'model_name': 'Dave2V1',
        'retrain_type': 'type1',
         'test_set': 'hybrid',
         'dataset': 'Udacity' #DAVE, Udacity
    },
]

TYPE1 = 'type1'
TYPE2 = 'type2'
HYBRID = 'hybrid'
ORIGINAL = 'original'
DAVE = 'DAVE'
UDACITY = 'UDACITY'
INF = 1e9
TYPE2_TEST_LIMIT = 300
lr = 1e-4
BATCH_SIZE = 128
SEED = 23456


def write_to_csv(output_file, data):
    FIELDS = ['model', 'dataset', 'retrain_type', 'test_set', 'ood_operator', 'selection_metric', 'epochs', 'id_ratio',
              'budget', 'original_acc', 'retrain_acc', 'acc_improvement', 'gd_num', 'w_gini', 'w_ent', 'ratio']
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

def get_max(df, filters):
    for k, v in filters.items():
        df = df[df[k] == v]
    if not df.empty:
        return df['acc_improvement'].max()
    else:
        return -INF

def retrain_selected(params, save=False):
    model_name = params['model_name']
    retrain_type = params['retrain_type'] #'type1'
    test_set = params['test_set']#'hybrid'
    dataset = params['dataset'] # 'DAVE', 'Udacity'
    hyper_param = {'dataset': dataset, 'random_seed': SEED, 'learning_rate': lr, 'batch_size': BATCH_SIZE,
                   'optimizer': 'adam'}
    # seed: 23456
    targeted_model_names_dict = model_scope_dict.copy()
    targeted_model = targeted_model_names_dict[model_name](hyper_params=hyper_param, mode='test')

    project_root = config.get('DEFAULT', 'project_root')
    if model_name == 'Dave2V1':
        adv_samples = '/adv-self-driving/attacker/adv_samples/Dave2v1'
        ori_samples = '/adv-self-driving/attacker/ori_samples/Dave2v1'
    else:
        adv_samples = os.path.join('/driving/attacker/adv_samples', targeted_model.model_name)
        ori_samples = os.path.join('/driving/attacker/ori_samples', targeted_model.model_name)

    log_path = os.path.join(project_root, 'output', 'our_method',
                            '{}_{}_{}_{}_{}_{}.csv'.format(m, model_name, retrain_type, test_set, dataset, lr,
                                                           our_method_params['ratio']))

    if dataset == 'DAVE':
        trainX, trainy, valX, valy, testX, testy, _ = DaveDataset(no_generator=True)
    elif dataset == 'Udacity':
        _, (trainX, trainy) = load_train_data(batch_size=hyper_param['batch_size'])
    perturbed_paths = {
        'FGSM': os.path.join(adv_samples, 'FGSM', dataset),
        'BIM': os.path.join(adv_samples, 'BIM', dataset),
        'Mementum': os.path.join(adv_samples, 'Momentum', dataset),
        'PGD': os.path.join(adv_samples, 'PGD', dataset)
    }
    ori_paths = {
        'FGSM': os.path.join(ori_samples, 'FGSM', dataset),
        'BIM': os.path.join(ori_samples, 'BIM', dataset),
        'Mementum': os.path.join(ori_samples, 'Momentum', dataset),
        'PGD': os.path.join(ori_samples, 'PGD', dataset)
    }

    # print('retraining with', params)
    selection_metric = params['selection_metric']
    bg = params['budget']
    id_dist = params['id_ratio']
    methods = params['method']
    epochs = retrain_type == TYPE1 and [30] or [5]

    for method in methods:
        perturbed_path = perturbed_paths[method]
        ori_path = ori_paths[method]

        idX, idy = LabeledDaveDataset(
            input_file=os.path.join(perturbed_path, 'data.txt'),
            path=ori_path + '/'
        )

        oodX, oody = LabeledDaveDataset(
            input_file=os.path.join(perturbed_path, 'data.txt'),
            path=perturbed_path + '/'
        )

        for budget in bg:
            for metric in selection_metric:
                _id_dist = id_dist
                if metric == 'dat' and len(id_dist) == 11:
                    _id_dist = id_dist[:-1] # remove 1.0

                accuracies, ori_accs = [], []
                for id_ratio in _id_dist:
                    #save_path = os.path.join(test_suite_dir, dataset, model_name, method, metric,
                    #                         str(our_method_params['w_gini']) + '_' + str(
                    #                             our_method_params['w_ent']) + '_' + str(our_method_params['gd_num']),
                    #                         str(budget), str(id_ratio))
                    resaved_suite_dir = '/adv-self-driving/repred_test_suite'
                    save_path = os.path.join(resaved_suite_dir, dataset, model_name, method, metric,
                                                                 str(budget), str(id_ratio))
                    if os.path.exists(os.path.join(save_path, 'data.txt')):
                        print(save_path, 'path exists')
                        continue

                    id_size = int(len(idX) * id_ratio)
                    ood_size = int(len(idX) - id_size)
                    
                    id_feat, id_y = idX[:id_size], idy[:id_size]
                    id_feat_can, id_y_can = id_feat[:int(id_size*0.5)], id_y[:int(id_size*0.5)]
                    id_feat_test, id_y_test = id_feat[int(id_size*0.5):], id_y[int(id_size*0.5):]

                    ood_feat, ood_y = oodX[id_size:], oody[id_size:]
                    ood_feat_can, ood_y_can = ood_feat[:int(ood_size*0.5)], ood_y[:int(ood_size*0.5)]
                    ood_feat_test, ood_y_test = ood_feat[int(ood_size*0.5):], ood_y[int(ood_size*0.5):]

                    hybrid_candidateX = np.concatenate((id_feat_can, ood_feat_can), axis=0)
                    hybrid_candidatey = np.concatenate((id_y_can, ood_y_can), axis=0)

                    hybrid_testX = np.concatenate((id_feat_test, ood_feat_test), axis=0)
                    hybrid_testy = np.concatenate((id_y_test, ood_y_test), axis=0)

                    _testX, _testy = hybrid_testX, hybrid_testy

                    if test_set == ORIGINAL:
                        _testX, _testy = testX, testy
                    
                    ori_acc = targeted_model.test(testX=_testX, testy=_testy)

                    retrainX, retrainy = None, None

                    if budget == 1.0:
                        if retrain_type == TYPE1:
                            retrainX, retrainy = hybrid_candidateX, hybrid_candidatey
                        else:
                            trun = min(len(trainX), TYPE2_TEST_LIMIT)
                            retrainX = np.concatenate((hybrid_candidateX, trainX[:trun]), axis=0)
                            retrainy = np.concatenate((hybrid_candidatey, trainy[:trun]), axis=0)
                    else:

                        selected_candidateX, selected_candidatey = targeted_model.selection(
                            budget=budget,
                            trainX=trainX[:TYPE2_TEST_LIMIT],
                            trainy=trainy[:TYPE2_TEST_LIMIT],
                            candidateX=hybrid_candidateX,
                            candidateX_id=id_feat_can,
                            candidatey=hybrid_candidatey,
                            candidatey_id=id_y_can,
                            hybrid_test=hybrid_testX,
                            hybrid_testy=hybrid_testy,
                            metric=metric,
                            id_ratio=id_ratio,
                            our_method_params=our_method_params
                        )
                    if save:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        selected_pred = targeted_model.test_pred(selected_candidateX, selected_candidatey)
                        save_path = os.path.join(save_path, 'data.txt')
                        for idx, (X, y_gt, y_pred) in enumerate(zip(selected_candidateX, selected_candidatey, selected_pred)):
                            with open(save_path, 'a') as f:
                                f.write('{} {} {}\n'.format(X, int(y_gt), int(y_pred)))

                        if retrain_type == TYPE1:
                            retrainX, retrainy = selected_candidateX, selected_candidatey
                        
                        else:
                            trun = min(len(trainX), TYPE2_TEST_LIMIT)
                            retrainX = np.concatenate((selected_candidateX, trainX[:trun]), axis=0)
                            retrainy = np.concatenate((selected_candidatey, trainy[:trun]), axis=0)
                    
                    retrain_acc = targeted_model.retrain(candidateX=retrainX, candidatey=retrainy, testX=_testX, testy=_testy, epochs=epochs)
                    accuracies.append(retrain_acc*100)
                    ori_accs.append(ori_acc*100)

                    write_to_csv(log_path, [
                        {
                            'model': model_name,
                            'dataset': dataset,
                            'retrain_type': retrain_type,
                            'test_set': test_set,
                            'ood_operator': method,
                            'selection_metric': metric,
                            'epochs': epochs,
                            'id_ratio': id_ratio,
                            'budget': budget,
                            'original_acc': ori_acc * 100,
                            'retrain_acc': retrain_acc * 100,
                            'acc_improvement': (retrain_acc - ori_acc) * 100,
                            'gd_num': our_method_params['gd_num'],
                            'w_gini': our_method_params['w_gini'],
                            'w_ent': our_method_params['w_ent'],
                            'ratio': our_method_params['ratio']
                        }
                    ])






def main():
    for test in tests:
         retrain_selected(test, True)

if __name__ == '__main__':
    main()

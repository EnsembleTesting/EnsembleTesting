'''
Driving models DAVE2V1
'''
import sys
sys.path.append('/adv-self-driving')
from tools import utils
from classification import *
from dataloader.data_utils import *
from config import config
from timeit import default_timer
from datetime import datetime
import os
from tqdm import tqdm
import time
from metrics import *
tf.debugging.set_log_device_placement(True)

class Dave2v1(Classifier):
    def __init__(self,
                 hyper_params=None,
                 reuse=False,
                 is_saving=True,
                 init_graph=True,
                 mode='train',
                 name='Dave2v1'):
        """
        build Epoch driving model
        @param info_dict: None,
        @param hyper_params: hyper parameters,
        @param reuse: reuse the variables or not
        @param is_saving: option for saving weights
        @param init_graph: initialize graph
        @param mode: enable a mode for run the model, 'train' or 'test'
        @param name: model name
        """
        super(Dave2v1, self).__init__()
        self.is_saving = is_saving
        self.init_graph = init_graph
        try:
            assert mode == 'train' or mode == 'test'
        except:
            raise AssertionError("'train' or 'test' mode, not others.")

        self.mode = mode
        self.hp_params_dict = hyper_params
        self.hp_params = utils.ParamWrapper(self.hp_params_dict)
        self.model_name = name
        self.input_dim = (100,100,3)
        if self.is_saving:
            self.save_dir = config.get('experiments', name.lower()) + '_' + self.hp_params.dataset
            self.save_dir_retrain = config.get('experiments', name.lower()) + '_' + self.hp_params.dataset + '_retrain'
        tf.set_random_seed(self.hp_params.random_seed)
        if self.init_graph:
            self.model_graph(reuse=reuse)

    def model_graph(self, reuse=False):
        """
        build the graph
        """
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='X')
        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None, ], name='Y')
        self.is_training = tf.placeholder(tf.bool, name="TRAIN")

        tf.set_random_seed(self.hp_params.random_seed)
        self.softmax, self.y_tensor, _, _ = self.forward(self.x_input, self.y_input, reuse=reuse)
        self.model_inference()

    def graph(self, x_input,
              name="Epoch",
              reuse=False):
        with tf.variable_scope("{}".format(name), reuse=reuse):
            bn1 = tf.layers.batch_normalization(x_input,  epsilon=0.001, momentum=0.99)
            conv1 = tf.layers.conv2d(bn1, filters=24, kernel_size=5, padding='valid', activation=tf.nn.relu, strides=(2,2)) #
            print(conv1.shape)
            conv2 = tf.layers.conv2d(conv1, filters=36, kernel_size=5, padding='valid', activation=tf.nn.relu, strides=(2,2)) #
            conv3 = tf.layers.conv2d(conv2, filters=48, kernel_size=5, padding='valid', activation=tf.nn.relu, strides=(2,2))
            conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=3, padding='valid', activation=tf.nn.relu, strides=(1,1))
            conv5 = tf.layers.conv2d(conv4, filters=64, kernel_size=3, padding='valid', activation=tf.nn.relu, strides=(1,1)) # (5,5,64)

            dense1 = tf.layers.dense(inputs=tf.layers.Flatten()(conv5), units=582, activation=tf.nn.relu, name="DENSE1")
            dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.relu, name="DENSE2")
            dense3 = tf.layers.dense(inputs=dense2, units=50, activation=tf.nn.relu, name="DENSE3")
            dense4 = tf.layers.dense(inputs=dense3, units=10, activation=tf.nn.relu, name="DENSE4")
            dense5 = tf.layers.dense(inputs=dense4, units=3, activation=tf.nn.softmax, name="DENSE5")

            return dense5, conv1, dense1

    def forward(self, x_tensor, y_tensor, reuse=False):
        """
        let data pass through the neural network
        :param x_tensor: input data
        :type: Tensor.float32
        :param y_tensor: label
        :type: Tensor.int64
        :param reuse: Boolean
        :return: Null
        """
        self.nn = self.graph
        softmax, self.conv1, self.dense1 = self.graph(
            x_tensor, name=self.model_name, reuse=reuse
        )
        return softmax, y_tensor, None, None

    def model_inference(self):
        """
        model inference
        """
        # loss definition
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor,
            logits=self.softmax)

        # prediction
        self.y_pred = tf.argmax(self.softmax, axis=1)
        self.y_proba = self.softmax

        # some information
        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred, self.y_tensor)))

    def train(self, train_generator=None, val_xs=None, val_ys=None):
        """train dnn"""
        global_train_step = tf.train.get_or_create_global_step()
        saver = tf.train.Saver()
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()

        # optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,
                                                                                      global_step=global_train_step)
        tf_cfg = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        tf_cfg.gpu_options.allow_growth = True
        tf_cfg.gpu_options.per_process_gpu_memory_fraction = 1.
        sess = tf.Session(config=tf_cfg)
        with sess.as_default():
            summary_writer = tf.summary.FileWriter(self.save_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            training_time = 0.0
            output_steps = 20
            best_acc = 0.
            step_idx = 0
            for X_batch, y_batch in tqdm(train_generator):

                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx + 1) % output_steps == 0:
                    validation_generator = val_generator(val_xs, val_ys,
                                  batch_size=self.hp_params.batch_size)
                    print('Step {}:{}'.format(step_idx + 1, datetime.now()))
                    val_res_list = [sess.run([self.accuracy, self.y_pred], feed_dict={self.x_input: valX_batch,
                                                                                      self.y_input: valy_batch,
                                                                                      self.is_training: False}) \
                                    for [valX_batch, valy_batch] in tqdm(validation_generator)
                                    ]
                    val_res = np.array(val_res_list, dtype=object)
                    _acc = np.mean(val_res[:, 0])
                    _pred_y = np.concatenate(val_res[:, 1])

                    if step_idx != 0:
                        print('    {} samples per second'.format(
                            output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    dense4, conv5, y_pred, y_label, softmax, train_acc, train_loss = sess.run([self.dense4, self.conv5, self.y_pred, self.y_tensor, self.softmax, self.accuracy, self.cross_entropy], feed_dict=train_dict)
                    vars_val = sess.run(tf.trainable_variables())

                    if best_acc < _acc:
                        best_acc = _acc
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        print('    Saving models to', os.path.join(self.save_dir, 'checkpoint'))
                        saver.save(sess,
                                   os.path.join(self.save_dir, 'checkpoint'),
                                   global_step=global_train_step)

                start = default_timer()
                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
                step_idx += 1
                if step_idx > 50000:
                    print('finish training')
                    break
        sess.close()

    def test(self, testX=None, testy=None):
        test_generator = val_generator(testX, testy,
                                             batch_size=self.hp_params.batch_size)
        self.mode = 'test'

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)

        if cur_checkpoint is None:
            print("No saved parameters")
            return
        # load parameters
        saver = tf.train.Saver()
        eval_dir = os.path.join(self.save_dir, 'eval')
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            with sess.as_default():
                pred = []
                for X_batch, y_batch in tqdm(test_generator):
                    test_dict = {
                        self.x_input: X_batch,
                        self.y_input: y_batch,
                        self.is_training: False
                    }
                    _y_pred = sess.run(self.y_pred, feed_dict=test_dict)
                    pred.append(_y_pred)
                y_pred = np.concatenate(pred)
                #import pdb; pdb.set_trace()

            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            accuracy = accuracy_score(testy, y_pred)
            b_accuracy = balanced_accuracy_score(testy, y_pred)

            MSG = "The accuracy on the test dataset is {:.5f}%"
            print(MSG.format(accuracy * 100))
            MSG = "The balanced accuracy on the test dataset is {:.5f}%"
            print(MSG.format(b_accuracy * 100))
            sess.close()

        return accuracy

    def test_pred(self, testX=None, testy=None):
        test_generator = val_generator(testX, testy,
                                       batch_size=self.hp_params.batch_size)
        self.mode = 'test'
        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('****************In test_rpst. Loading saved parameters from', self.save_dir)
        if cur_checkpoint is None:
            print("No saved parameters")
            return
        # load parameters
        saver = tf.train.Saver()
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            with sess.as_default():
                pred = []
                for X_batch, y_batch in tqdm(test_generator):
                    test_dict = {
                        self.x_input: X_batch,
                        self.y_input: y_batch,
                        self.is_training: False
                    }
                    _y_pred = sess.run(self.y_pred, feed_dict=test_dict)
                    pred.append(_y_pred)
                y_pred = np.concatenate(pred)
            sess.close()
        return y_pred

    def retrain(self, candidateX=None, candidatey=None, testX=None, testy=None, epochs=None):
        self.hp_params.n_epochs = epochs
        # use candidate set for retraining
        train_generator = data_generator(candidateX, candidatey,
                                     batch_size=self.hp_params.batch_size)

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        # load parameters
        saver = tf.train.Saver()
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(self.save_dir_retrain, sess.graph)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()

        # optimizer
        global_train_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,global_step=global_train_step)

        with sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, cur_checkpoint)

            training_time = 0.0
            # train_input.reset_cursor()
            step_idx = 0
            best_acc = 0.
            output_steps = 1
            for X_batch, y_batch in tqdm(train_generator):
                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx) % output_steps == 0:
                    print('        Validating Retraining......')
                    validation_generator = val_generator(testX, testy,
                                                         batch_size=self.hp_params.batch_size)
                    # print('Step {}/{}:{}'.format(step_idx + 1, train_input.steps, datetime.now()))
                    # val_input.reset_cursor()
                    val_res_list = [sess.run([self.accuracy, self.y_pred], feed_dict={self.x_input: valX_batch,
                                                                                      self.y_input: valy_batch,
                                                                                      self.is_training: False}) \
                                    for [valX_batch, valy_batch] in tqdm(validation_generator)
                                    ]
                    val_res = np.array(val_res_list, dtype=object)
                    _acc = np.mean(val_res[:, 0])
                    _pred_y = np.concatenate(val_res[:, 1])

                    if step_idx != 0:
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))

                    if best_acc < _acc:
                        best_acc = _acc


                start = default_timer()
                step_idx += 1
                epoch = int((step_idx * self.hp_params.batch_size)/len(candidateX))
                if epoch > epochs[0]:
                    print('finish retraining, step idx = ', step_idx, 'retrain epochs = ', epochs, 'best acc', best_acc)
                    break
                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()
        return best_acc

    def selection(self, budget=None, trainX=None, trainy=None, candidateX=None, candidateX_id=None, candidatey=None, candidatey_id=None, hybrid_test=None, hybrid_testy=None, metric=None, id_ratio=None, our_method_params=None):
        # select from hybrid candidate set
        self.mode = 'test'
        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('****************In test_rpst. Loading saved parameters from', self.save_dir)
        # load parameters
        saver = tf.train.Saver()
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            if metric == 'random':
                selected_candidateX, selected_candidatey = Random(sess, candidateX, candidatey, budget)
            elif metric == 'deepgini':
                selected_candidateX, selected_candidatey, _ = deepgini(sess, candidateX, candidatey, self, budget)
            elif metric == 'entropy':
                selected_candidateX, selected_candidatey, _ = entropy(sess, candidateX, candidatey, self, budget)
            elif metric == 'gd':
                selection_size = int(budget * candidateX.shape[0])
                selected_candidateX, selected_candidatey = GD(sess, candidateX, candidatey, self, selection_size, 60)
            elif metric == 'dsa':
                selected_candidateX, selected_candidatey = dsa_select(sess, trainX, trainy, candidateX, candidatey,
                                                                      self, budget, dataset, self.model_name)
            elif metric == 'deepgd':
                selected_candidateX, selected_candidatey = deepGD(sess, candidateX, candidatey, self, budget)
            elif metric == 'ensemble_simple':
                selected_candidateX, selected_candidatey = ensemble_simple(sess, candidateX, candidatey, self, budget, our_method_params)
            elif metric == 'ensemble_simple_add':
                selected_candidateX, selected_candidatey = ensemble_simple_add(sess, candidateX, candidatey, self, budget, our_method_params)
            elif metric == 'ensemble_p_values_fisher':
                selected_candidateX, selected_candidatey = ensemble_p_values_fisher(sess, trainX, trainy, candidateX, candidatey, self, budget, our_method_params)
            elif metric == 'dat_ood_detector':
                selected_candidateX, selected_candidatey = dat_ood_detector(sess, trainX, trainy, candidateX, candidatey,
                                                                     hybrid_test,
                                                                    hybrid_testy, self,
                                                                     budget)
            sess.close()
        return selected_candidateX, selected_candidatey

def train_udacity():
    hyper_param = {'dataset': 'Udacity', 'random_seed': 23456, 'learning_rate': 1e-5, 'batch_size': 128, 'optimizer': 'adam'}
    # load train/test generator:
    train_generator, num_train = load_train_data(batch_size=hyper_param['batch_size'])
    # test_generator, _ = load_test_data(batch_size=hyper_param['batch_size'])
    val_xs, val_ys = load_val_data(batch_size=hyper_param['batch_size'], start=20000, end=24000)
    print('number of train data', num_train, 'number of val data', len(val_xs))
    model = Dave2v1(hyper_params=hyper_param)
    model.train(train_generator, val_xs, val_ys)

def train_dave():
    hyper_param = {'dataset': 'DAVE', 'random_seed': 23456, 'learning_rate': 1e-5, 'batch_size': 128, 'optimizer': 'adam'}
    # load train/test generator:
    train_generator, val_xs, val_ys, test_xs, test_ys, steering_values = DaveDataset(batch_size=hyper_param['batch_size'])

    model = Dave2v1(hyper_params=hyper_param)
    model.train(train_generator, val_xs, val_ys)

def test_udacity():
    hyper_param = {'dataset': 'Udacity', 'random_seed': 23456, 'learning_rate': 1e-5, 'batch_size': 128,
                   'optimizer': 'adam'}
    #train_generator, num_train = load_train_data(batch_size=hyper_param['batch_size'])
    testX, testy = load_val_data(batch_size=hyper_param['batch_size']) #5614 test data
    import pdb; pdb.set_trace()
    model = Dave2v1(hyper_params=hyper_param)
    model.test(testX, testy)

def test_dave():
    hyper_param = {'dataset': 'DAVE', 'random_seed': 23456, 'learning_rate': 1e-5, 'batch_size': 128, 'optimizer': 'adam'}
    # load train/test generator:
    train_generator, val_xs, val_ys, test_xs, test_ys, steering_values = DaveDataset(batch_size=hyper_param['batch_size'])

    model = Dave2v1(hyper_params=hyper_param)
    model.test(test_xs, test_ys)
if __name__ == '__main__':

    test_dave()



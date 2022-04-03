from __future__ import print_function

import numpy as np
# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import csv
import os
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import argparse

from binary_connect import binary_connect
from binary_connect.utils import print_weight_counts
from crn.utils import translate_to_crn
from crn.utils import translate_examples
from crn.utils import save_model, load_model
from crn.constants import virus_dataset_shuffled_indices, virus_example_indices
from crn.constants import SIMULATION_RES_DIR
from crn.hyperparams import HyperparamsConfig
from crn.hyperparams import compute_LR_decay
from binary_connect.mlp_model import make_model


hyperparams = HyperparamsConfig(
    BATCH_SIZE=16,
    ALPHA=.15,
    EPSILON=1e-4,
    USE_BATCH_NORMALIZATION=False,
    HIDDEN_LAYERS_DIMS=[8],
    NUM_EPOCHS=500,
    DROPOUT_IN=0.,
    DROPOUT_HIDDEN=0.,
    BINARIZATION='TERNARY',
    ZERO_THRESHOLD=0.15,
    STOCHASTIC=False,
    # (-H,+H) are the two binary values
    H=1.,
    W_LR_scale="Glorot", # "Glorot" means we are using the coefficients from Glorot's paper
    LR_start=0.01,
    LR_fin=3e-5,
    LR_decay_type='exponential' # (LR_fin / LR_start) ** (1. / NUM_EPOCHS)
)

INPUT_DIMS = 10

# One hot representation.
OUTPUT_DIMS = 4

MODEL_FILE = '../data-repo/models/virus.pkl'


class Model(object):
    def __init__(self, hyperparams):
        # for reproducibility
        np.random.seed(1234)
        self.hyperparams = hyperparams
        result = make_model(hyperparams,
                            INPUT_DIMS,
                            OUTPUT_DIMS,
                            make_dropout_layer_when_zero=True)
        self.train_fn = result['train_fn']
        self.val_fn = result['val_fn']
        self.test_output = result['test_output']
        self.input = result['input']
        self.mlp = result['mlp']

    def nn_output(self, x):
        """Output of network for a given input."""
        return (theano.function(
            [self.input], [self.test_output])([x]))[0][0]

    names = ["Linear Model", "Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net",
             "AdaBoost", "Naive Bayes", "QDA"]

    classifiers = [
        linear_model.SGDClassifier(),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    def load_dataset(self):
        gene_names = []
        sample_names = []
        rows = []

        subject_dict = {}

        for config_name in ['DEE3', 'DEE4', 'DEE1', 'DEE2', 'DEE5', 'UVA', 'DUKE']:
            with open('../data-repo/datasets/virus/icml2020dataset_GSE73072_{}.csv'.format(config_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    line_count += 1
                    if line_count == 1:
                        pass
                    else:
                        subject, sham, symptomatic, infected, onset_time, offset_time = row
                        symptomatic = symptomatic == '1'
                        infected = infected == '1'
                        if infected:
                            onset_time = float(onset_time)
                            offset_time = float(offset_time)
                        else:
                            onset_time = 0
                            offset_time = 0
                        key = config_name + '#' + subject
                        if key in subject_dict:
                            raise Exception('')
                        subject_dict[key] = [infected, onset_time, offset_time]

        sample_dict = {}

        with open('../data-repo/datasets/virus/icml2020dataset_GSE73072_sample.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            sample_info = []
            for row in csv_reader:
                line_count += 1
                if line_count == 1:
                    pass
                else:
                    sample_name = row[0]
                    try:
                        disease_name, config, _, subject, _, hour = row[1].split()
                    except ValueError:
                        # This is issue with 'DEE4' name -> 'DEE 4'.
                        disease_name, config1, config2, _, subject, _, hour = row[
                            1].split()
                        config = config1 + config2
                    subject = int(subject[:-1])
                    key = config + '#' + str(subject)
                    hour = float(hour)
                    if key in subject_dict:
                        infected, onset_time, offset_time = subject_dict[key]
                        # TODO: Check if onset and offset time should be used.
                        sample_dict[sample_name] = (infected
                                                    and hour >= onset_time
                                                    and hour < offset_time)

        with open('../data-repo/datasets/virus/icml2020dataset_GSE73072_4labels.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if line_count == 1:
                    sample_names = row[1:]
                else:
                    gene_names.append(row[0])
                    rows.append(row[1:])

        data_X = np.swapaxes(np.array(rows[:-1], np.float32), 0, 1)
        data_X_filtered = []
        data_Y_filtered = []
        for i in range(data_X.shape[0]):
            sample_name = sample_names[i]
            if sample_name in sample_dict:
                if sample_dict[sample_name]:
                    data_X_filtered.append(data_X[i])
                    data_Y_filtered.append(rows[-1][i])
                # else:
                #     data_Y_filtered.append(0)
        data_X = np.array(data_X_filtered)

        # NOTE:
        # Without scaling there are training issues
        # (when not using Batch Normalization).
        # Namely validation loss can significantly reduce,
        # while validation error grows.
        scaler = StandardScaler()
        scaler.fit(data_X)
        data_X = scaler.transform(data_X)

        # Labels are in the last row
        label_names = np.unique(np.array(rows[-1])).tolist()
        data_Y = np.array([label_names.index(e) for e in data_Y_filtered], np.int)

        data_Y = np.float32(np.eye(OUTPUT_DIMS)[data_Y])
        # for hinge loss
        # make y be in {-1, 1}.
        data_Y = 2 * data_Y - 1.

        # indices = np.arange(data_X.shape[0])
        # np.random.shuffle(indices)
        # Make sure to always use the same splits.
        indices = np.array(virus_dataset_shuffled_indices)

        num_train = int(0.80 * data_X.shape[0])
        num_valid = int(0.05 * data_X.shape[0])
        end_train = num_train
        start_valid = end_train + 1
        end_valid = start_valid + num_valid
        start_test = end_valid + 1
        train_X = data_X[indices[:end_train]]
        train_Y = data_Y[indices[:end_train]]
        valid_X = data_X[indices[start_valid:end_valid]]
        valid_Y = data_Y[indices[start_valid:end_valid]]
        test_X = data_X[indices[start_test:]]
        test_Y = data_Y[indices[start_test:]]
        print('Dataset size: {}'.format(data_X.shape[0]))
        print('Train set size: {}'.format(train_X.shape[0]))
        print('Valid set size: {}'.format(valid_X.shape[0]))
        print('Test set size: {}'.format(test_X.shape[0]))

        # print('gene names:'.format('; '.join(gene_names)))
        # print('label names: '.format('; '.join(label_names)))

        return train_X, train_Y, valid_X, valid_Y, test_X, test_Y, scaler

    def test_scikit_classifiers(self):
        train_X, train_Y, valid_X, valid_Y, test_X, test_Y, _ = self.load_dataset()

        # Testing accuracy of standard classifiers in this domain.
        for name, clf in zip(self.names, self.classifiers):
            clf.fit(train_X, np.argmax(train_Y, axis=1))
            score = clf.score(test_X, np.argmax(test_Y, axis=1))
            print("classifier: {}, test score: {}".format(name, score))
            score = clf.score(valid_X, np.argmax(valid_Y, axis=1))
            print("classifier: {}, val score: {}".format(name, score))

    def translate(self, test_X, test_Y, scaler):
        """
        :param scaler: Scaler used to transform original features.
        """
        print_weight_counts(self.mlp)
        final_loss, final_err = self.val_fn(test_X, test_Y)
        print('final loss (test): ' + str(final_loss))
        print('final error (test): ' + str(final_err * 100) + '%')

        num_examples = 100
        np.random.seed(69)
        examples_indices = np.random.choice(test_X.shape[0], num_examples,
                                            replace=False)
        # TODO:
        # examples_indices = disease_example_indices
        examples_X = test_X[examples_indices]
        examples_Y = test_Y[examples_indices]
        # model_output1 = theano.function([self.input], [self.test_output])(
        #     examples_X)
        # model_output2 = theano.function([self.input],
        #                                 [T.argmax(self.test_output, axis=1)])(
        #     examples_X)
        # print("Outputs of a trained model:")
        # print(model_output1)
        # print(model_output2)
        #
        # print("Correct Outputs:")
        # print(examples_Y)
        # print(np.argmax(examples_Y, axis=1))
        #
        # simulate_network(self.mlp, examples_X)

        for shift_inputs in [False, True]:
            # shift_inputs: Whether to shift inputs so they are always positive.
            # This optimizes input layer so it doesn't require
            # dual-rail representation.

            if shift_inputs:
                mathematica_file = '../data-repo/mathematica/virus-inputs-shifted.wls'
                results_dir = os.path.join(SIMULATION_RES_DIR,
                                           'virus-inputs-shifted')
                # 0 value is the minimum value of the gene expression.
                # can't get below that.
                inputs_shift = -scaler.transform([[0.] * test_X.shape[1]])[0]
            else:
                mathematica_file = '../data-repo/mathematica/virus.wls'
                results_dir = os.path.join(SIMULATION_RES_DIR, 'virus')
                inputs_shift = None

            with open(mathematica_file, 'w') as f:
                translate_to_crn(self.mlp, 'VIRUS', f, inputs_shift)
                translate_examples('VIRUS',
                                   examples_X,
                                   examples_Y,
                                   self.nn_output,
                                   last_layer_num=len(self.hyperparams.HIDDEN_LAYERS_DIMS),
                                   f=f,
                                   results_dir=results_dir,
                                   inputs_shift=inputs_shift)
                translate_examples('VIRUSReduced',
                                   examples_X,
                                   examples_Y,
                                   self.nn_output,
                                   last_layer_num=len(self.hyperparams.HIDDEN_LAYERS_DIMS),
                                   f=f,
                                   results_dir=results_dir,
                                   inputs_shift=inputs_shift)
    
    def train(self, save_files=True):
        train_X, train_Y, valid_X, valid_Y, test_X, test_Y, _ = self.load_dataset()

        print('Training...')

        lr_decay = compute_LR_decay(LR_start=self.hyperparams.LR_start,
                                    LR_fin=self.hyperparams.LR_fin,
                                    NUM_EPOCHS=self.hyperparams.NUM_EPOCHS,
                                    LR_decay_type=self.hyperparams.LR_decay_type
                                    )

        result = binary_connect.train(
            self.train_fn,
            self.val_fn,
            self.hyperparams.BATCH_SIZE,
            self.hyperparams.LR_start, lr_decay,
            self.hyperparams.NUM_EPOCHS,
            X_train=train_X, y_train=train_Y,
            X_val=valid_X, y_val=valid_Y,
            X_test=test_X, y_test=test_Y,
            network=self.mlp,
            return_best_epoch=True,
            callback_after_epoch=None)
        print("Finished Training")

        print_weight_counts(self.mlp)

        if save_files:
            save_model(self, MODEL_FILE)

        # training_losses = result['training_losses']
        # validation_losses = result['validation_losses']
        # plt.plot(np.arange(len(training_losses)), training_losses)
        # plt.xlabel('epoch')
        # plt.ylabel('loss (train set)')
        # if save_files:
        #     plt.savefig(out_dir + '/train_loss.pdf')
        #     plt.close()
        # plt.plot(np.arange(len(validation_losses)), validation_losses)
        # plt.xlabel('epoch')
        # plt.ylabel('loss (validation set)')
        # if save_files:
        #     plt.savefig(out_dir + '/val_loss.pdf')
        #     plt.close()

        return result

    def lr_rate_range_test(self):
        train_X, train_Y, valid_X, valid_Y, test_X, test_Y, _ = self.load_dataset()

        print('Training...')

        start_lr = 1e-6
        end_lr = 1e-1
        num_epochs = 100
        decay = (end_lr / start_lr) ** (1. / num_epochs)

        learning_rates = []
        current_lr = start_lr
        for i in range(num_epochs):
            learning_rates.append(current_lr)
            current_lr *= decay

        result = binary_connect.train(
            self.train_fn,
            self.val_fn,
            self.hyperparams.BATCH_SIZE,
            start_lr, decay,
            num_epochs,
            X_train=train_X, y_train=train_Y,
            X_val=valid_X, y_val=valid_Y,
            X_test=test_X, y_test=test_Y,
            network=self.mlp,
            return_best_epoch=True)

        training_losses = result['training_losses']
        validation_losses = result['validation_losses']

        print("Finished Training")

        plt.plot(learning_rates, training_losses)
        plt.xlabel('LR')
        plt.xscale('log')
        plt.ylabel('loss (train set)')
        plt.show()

        plt.plot(learning_rates, validation_losses)
        plt.xlabel('LR')
        plt.xscale('log')
        plt.ylabel('loss (val set)')
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr_range_test', action='store_true')
    parser.add_argument('--scikit_classifiers_test', action='store_true')
    parser.add_argument('--translate', action='store_true')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_units', type=int)
    parser.add_argument('--start_lr', type=float)
    parser.add_argument('--end_lr', type=float)
    parser.add_argument('--dropout_in', type=float)
    parser.add_argument('--dropout_hidden', type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.num_epochs is not None:
        hyperparams.NUM_EPOCHS = args.num_epochs
    if args.start_lr is not None:
        hyperparams.LR_start = args.start_lr
    if args.end_lr is not None:
        hyperparams.LR_fin = args.end_lr
    if args.num_layers is not None:
        hyperparams.HIDDEN_LAYERS_DIMS = np.repeat(
            np.array([args.num_units]), args.num_layers).tolist()
    if args.dropout_in is not None:
        hyperparams.DROPOUT_IN = args.dropout_in
    if args.dropout_hidden is not None:
        hyperparams.DROPOUT_HIDDEN = args.dropout_hidden
    if args.lr_range_test:
        Model(hyperparams).lr_rate_range_test()
    if args.scikit_classifiers_test:
        Model(hyperparams).test_scikit_classifiers()
    if args.train:
        Model(hyperparams).train(save_files=True)
    if args.translate:
        model = load_model(MODEL_FILE)
        _, _, _, _, test_X, test_Y, scaler = model.load_dataset()
        # print("Hyperparams:")
        # print(model.hyperparams)
        model.translate(test_X, test_Y, scaler)

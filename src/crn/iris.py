from __future__ import print_function

import numpy as np
# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T
import argparse
from pylearn2.datasets.iris import Iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from binary_connect import binary_connect
from crn.utils import simulate_network
from crn.utils import translate_to_crn
from crn.utils import translate_examples
from crn.utils import save_model, load_model
from binary_connect.utils import print_weight_counts
from crn.hyperparams import HyperparamsConfig
from crn.hyperparams import compute_LR_decay
from binary_connect.mlp_model import make_model

import os
from constants import SIMULATION_RES_DIR

INPUT_DIMS = 4
# One hot representation.
OUTPUT_DIMS = 3

MODEL_FILE = '../data-repo/models/iris.pkl'

hyperparams = HyperparamsConfig(
    BATCH_SIZE=16,
    ALPHA=.15,
    EPSILON=1e-4,
    USE_BATCH_NORMALIZATION=False,
    HIDDEN_LAYERS_DIMS=[3],
    NUM_EPOCHS=10000,
    DROPOUT_IN=0.1,
    DROPOUT_HIDDEN=0.01,
    BINARIZATION='TERNARY',
    ZERO_THRESHOLD=0.15,
    STOCHASTIC=False,
    # (-H,+H) are the two binary values
    H=1.,
    # "Glorot" means we are using the coefficients from Glorot's paper
    W_LR_scale="Glorot",
    LR_start=0.02,
    LR_fin=0.02,
    LR_decay_type='exponential' # (LR_fin / LR_start) ** (1. / NUM_EPOCHS)
)


class Model(object):

    def __init__(self, hyperparams):
        np.random.seed(1234)  # for reproducibility
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

    def load_dataset(self):
        dataset = Iris()
        original_train_X = dataset.X

        scaler = StandardScaler()
        scaler.fit(original_train_X)

        train_X = scaler.transform(original_train_X)
        train_Y = dataset.y

        train_Y = np.hstack(train_Y)
        train_Y = np.float32(np.eye(OUTPUT_DIMS)[train_Y])

        # for hinge loss
        # make y be in {-1, 1}.
        train_Y = 2 * train_Y - 1.

        return train_X, train_Y

    def translate(self, train_X, train_Y):
        print_weight_counts(self.mlp)

        final_loss, final_err = self.val_fn(train_X, train_Y)
        print()
        print('Final loss: {:.2f}'.format(final_loss))
        print('Final error: {:.2f}%'.format(final_err * 100))
        print()

        # examples_indices = np.random.choice(train_X.shape[0], 5)
        examples_indices = [0, 50, 100]
        examples_X = train_X[examples_indices]
        examples_Y = train_Y[examples_indices]
        # model_output1 = theano.function([self.input], [self.test_output])(
        #     examples_X)
        # model_output2 = theano.function([self.input],
        #                                 [T.argmax(self.test_output, axis=1)])(
        #     examples_X)
        # print("Outputs of a trained model:")
        # print(model_output1)
        # print(model_output2)

        # print("Correct Outputs:")
        # print(examples_Y)
        # print(np.argmax(examples_Y, axis=1))

        # simulate_network(self.mlp, examples_X)

        num_layers = len(self.hyperparams.HIDDEN_LAYERS_DIMS)

        results_dir = os.path.join(SIMULATION_RES_DIR, 'iris')

        with open('../data-repo/mathematica/iris.wls', 'w') as f:
            translate_to_crn(self.mlp, 'IRIS', f)
            translate_examples('IRIS',
                               train_X,
                               train_Y,
                               self.nn_output,
                               last_layer_num=num_layers,
                               f=f,
                               tmax=50,
                               results_dir=results_dir)
            translate_examples('IRISReduced',
                               train_X,
                               train_Y,
                               self.nn_output,
                               last_layer_num=num_layers,
                               f=f,
                               tmax=50,
                               results_dir=results_dir)

    def train(self):
        train_X, train_Y = self.load_dataset()

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
            self.hyperparams.LR_start,
            lr_decay,
            self.hyperparams.NUM_EPOCHS,
            X_train=train_X, y_train=train_Y,
            X_val=train_X, y_val=train_Y,
            X_test=train_X, y_test=train_Y,
            network=self.mlp,
            return_best_epoch=True)

        print("Finished Training")

        print_weight_counts(self.mlp)

        save_model(self, MODEL_FILE)

        # training_losses = result['training_losses']
        # plt.plot(np.arange(len(training_losses)), training_losses)
        # plt.xlabel('epoch')
        # plt.ylabel('loss (train set)')
        # plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--translate', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.train and not args.translate:
        print('Provide cmd args')
    if args.train:
        Model(hyperparams).train()
    if args.translate:
        model = load_model(MODEL_FILE)
        model.translate(*model.load_dataset())

from __future__ import print_function

import numpy as np
# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T
from utils import simulate_network
from utils import translate_to_crn
from utils import translate_examples
from pylearn2.datasets.mnist import MNIST
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import math
import copy
import argparse

from crn.utils import save_model, load_model
from binary_connect import binary_connect
from binary_connect.utils import print_weight_counts
from crn.constants import mnist_example_indices
from crn.hyperparams import HyperparamsConfig
from crn.hyperparams import compute_LR_decay
from binary_connect.mlp_model import make_model


hyperparams = HyperparamsConfig(
    BATCH_SIZE=100,
    ALPHA=.15,
    EPSILON=1e-4,
    USE_BATCH_NORMALIZATION=False,
    HIDDEN_LAYERS_DIMS=[256, 256, 256],
    NUM_EPOCHS=250,
    DROPOUT_IN=0.,
    DROPOUT_HIDDEN=0.,
    BINARIZATION='BINARY',
    ZERO_THRESHOLD=0.15,
    STOCHASTIC=False,
    # (-H,+H) are the two binary values
    H=1.,
    # "Glorot" means we are using the coefficients from Glorot's paper
    W_LR_scale="Glorot",
    LR_start=1e-5,
    LR_fin=3e-6,
    LR_decay_type='exponential' # (LR_fin / LR_start) ** (1. / NUM_EPOCHS)
)

ORIGINAL_IMAGE_DIM = 28
# If scaled equal to original then no scaling is performed.
SCALED_IMAGE_DIM = 14

INPUT_DIMS = SCALED_IMAGE_DIM**2

# One hot representation.
OUTPUT_DIMS = 10

MODEL_FILE = '../data-repo/models/mnist.pkl'


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

    def show_image(self, image_flattened):
        image_dim = int(math.sqrt(image_flattened.shape[0]))
        image = image_flattened.reshape(image_dim, image_dim)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(image, # interpolation='bilinear',
                   cmap=cm.Greys_r)
        plt.show()

    def save_image(self, image_flattened, out_file, use_cv2=True):
        image_dim = int(math.sqrt(image_flattened.shape[0]))
        image = image_flattened.reshape(image_dim, image_dim)
        if use_cv2:
            # Note that pixel values have to be between 0 and 255 in cv2.
            cv2.imwrite(out_file, 255. * image)
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.imshow(image, # interpolation='bilinear',
                       cmap=cm.Greys_r)
            fig.savefig(out_file, bbox_inches='tight')
            plt.close()

    def scale_image(self, image_flattened):
        image = image_flattened.reshape(ORIGINAL_IMAGE_DIM, ORIGINAL_IMAGE_DIM)
        res = cv2.resize(image, dsize=(SCALED_IMAGE_DIM, SCALED_IMAGE_DIM),
                         interpolation=cv2.INTER_LINEAR)
        return res.reshape(SCALED_IMAGE_DIM**2)

    def scale_images(self, images_flattened):
        resized = []
        for i in range(images_flattened.shape[0]):
            resized.append(self.scale_image(images_flattened[i]))
        return np.array(resized)

    def nn_output(self, x):
        """Output of network for a given input."""
        return (theano.function(
            [self.input], [self.test_output])([x]))[0][0]
        # return (theano.function(
        #     [self.input], [T.argmax(self.test_output, axis=1)])([x]))[0][0]

    def load_dataset(self):
        # Centering is done for each feature (pixel) in the image separately;
        # by subtracting the pixel value by the mean value (across examples)
        # for that pixel.
        center_images = True
        train_set = MNIST(which_set='train', start=0, stop=50000,
                          center=center_images)
        valid_set = MNIST(which_set='train', start=50000, stop=60000,
                          center=center_images)
        test_set = MNIST(which_set='test', center=center_images)

        test_set_uncentered_unscaled = MNIST(which_set='test', center=False)
        test_set_uncentered_scaled = MNIST(which_set='test', center=False)

        if SCALED_IMAGE_DIM < ORIGINAL_IMAGE_DIM:
            train_set.X = self.scale_images(train_set.X)
            valid_set.X = self.scale_images(valid_set.X)
            test_set.X = self.scale_images(test_set.X)
            test_set_uncentered_scaled.X = self.scale_images(
                test_set_uncentered_scaled.X)

        # bc01 format
        # print(train_set.X.shape)
        # train_set.X = train_set.X.reshape(-1, 1, 28, 28)
        # valid_set.X = valid_set.X.reshape(-1, 1, 28, 28)
        # test_set.X = test_set.X.reshape(-1, 1, 28, 28)

        # flatten targets
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)

        # One-hot the targets
        train_set.y = np.float32(np.eye(OUTPUT_DIMS)[train_set.y])
        valid_set.y = np.float32(np.eye(OUTPUT_DIMS)[valid_set.y])
        test_set.y = np.float32(np.eye(OUTPUT_DIMS)[test_set.y])

        # for hinge loss
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.

        return {
            'train_set': train_set,
            'valid_set': valid_set,
            'test_set': test_set,
            'test_set_uncentered_unscaled': test_set_uncentered_unscaled,
            'test_set_uncentered_scaled': test_set_uncentered_scaled
        }

    def train(self, save_files=True):
        data = self.load_dataset()
        train_set = data['train_set']
        valid_set = data['valid_set']
        test_set = data['test_set']

        print('Training...')

        lr_decay = compute_LR_decay(LR_start=self.hyperparams.LR_start,
                                    LR_fin=self.hyperparams.LR_fin,
                                    NUM_EPOCHS=self.hyperparams.NUM_EPOCHS,
                                    LR_decay_type=self.hyperparams.LR_decay_type
                                    )

        results = binary_connect.train(
            self.train_fn,
            self.val_fn,
            self.hyperparams.BATCH_SIZE,
            self.hyperparams.LR_start, lr_decay,
            self.hyperparams.NUM_EPOCHS,
            X_train=train_set.X, y_train=train_set.y,
            X_val=valid_set.X, y_val=valid_set.y,
            X_test=test_set.X, y_test=test_set.y,
            network=self.mlp,
            return_best_epoch=True)

        if save_files:
            save_model(self, MODEL_FILE)

        # Testing.
        print("Finished Training")

        # plt.plot(np.arange(len(training_losses)), training_losses)
        # plt.show()

        # self.translate(test_set, original_test_set)

        return results

    def translate(self, data):
        test_set = data['test_set']
        test_set_uncentered_scaled = data['test_set_uncentered_scaled']
        test_set_uncentered_unscaled = data['test_set_uncentered_unscaled']

        print_weight_counts(self.mlp)
        examples_indices = np.array(mnist_example_indices)
        examples_X = test_set.X[examples_indices]
        examples_X_uncentered = test_set_uncentered_scaled.X[examples_indices]
        examples_X_unscaled = test_set_uncentered_unscaled.X[examples_indices]
        examples_Y = test_set.y[examples_indices]

        for i in range(examples_indices.shape[0]):
            self.save_image(examples_X_uncentered[i],
                            '../data-repo/results/MNIST-input-resized_{}.png'.format(i))
            self.save_image(examples_X_unscaled[i],
                            '../data-repo/results/MNIST-input-original_{}.png'.format(i))

        final_loss, final_err = self.val_fn(test_set.X, test_set.y)
        print('final loss (test): ' + str(final_loss))
        print('final error (test): ' + str(final_err * 100) + '%')

        examples_X_cut = examples_X[:3]
        examples_Y_cut = examples_Y[:3]
        model_output1 = theano.function([self.input], [self.test_output])(
            examples_X_cut)
        model_output2 = theano.function([self.input],
                                        [T.argmax(self.test_output, axis=1)])(
            examples_X_cut)
        print("Outputs of a trained model:")
        print(model_output1)
        print(model_output2)
        print("Correct Outputs:")
        print(examples_Y_cut)
        print(np.argmax(examples_Y_cut, axis=1))
        simulate_network(self.mlp, examples_X_cut)

        with open('../data-repo/mathematica/mnist.wls', 'w') as f:
            translate_to_crn(self.mlp, 'MNIST', f)
            translate_examples('MNIST',
                               examples_X,
                               examples_Y,
                               self.nn_output,
                               last_layer_num=len(
                                   self.hyperparams.HIDDEN_LAYERS_DIMS),
                               f=f,
                               tmax=200)
            translate_examples('MNISTReduced',
                               examples_X,
                               examples_Y,
                               self.nn_output,
                               last_layer_num=len(
                                   self.hyperparams.HIDDEN_LAYERS_DIMS),
                               f=f,
                               tmax=200)


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
        print("Hyperparams:")
        print(model.hyperparams)
        data = model.load_dataset()
        model.translate(data)

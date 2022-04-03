from __future__ import print_function

# TODO: Change how models are saved -- only saved model weights.

import numpy as np
# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from skimage.color import gray2rgb
import os
import math
import argparse
from sklearn.preprocessing import StandardScaler

from binary_connect import binary_connect
from crn.utils import translate_to_crn
from crn.utils import translate_examples
from crn.utils import save_model, load_model
from binary_connect.mlp_model import make_model
from crn.hyperparams import compute_LR_decay
from crn.pattern_configs import *
from binary_connect.utils import print_weight_counts
import lasagne

from constants import SIMULATION_RES_DIR


def load_image(path):
    return cv2.imread(path)


def show_image(img):
    cv2.imshow('image', img)
    
    # Maintain output window utill 
    # user presses a key 
    cv2.waitKey(0)
      
    # Destroying present windows on screen 
    cv2.destroyAllWindows()

    
def convert_to_grayscale(img):
    return rgb2gray(img)


class Model(object):
    def __init__(self, hyperparams, pattern_config):
        # for reproducibility
        np.random.seed(1234)
        self.hyperparams = hyperparams
        self.pattern_config = pattern_config
        self.coordinate_system = pattern_config.COORDINATE_SYSTEM
        self.color_system = pattern_config.COLOR_SYSTEM

        if self.pattern_config.COORDINATE_SYSTEM == CoordinateSystem.CARTESIAN_DOUBLE:
            inputs = 4
        else:
            inputs = 2

        if self.pattern_config.COLOR_SYSTEM == ColorSystem.BLACK_AND_WHITE:
            loss_func = 'HINGE'
            self.outputs = 2
        elif self.pattern_config.COLOR_SYSTEM == ColorSystem.GRAYSCALE:
            loss_func = 'MSE'
            self.outputs = 1

        result = make_model(hyperparams,
                            inputs,
                            self.outputs,
                            loss_func=loss_func,
                            make_dropout_layer_when_zero=True)
        self.train_fn = result['train_fn']
        self.val_fn = result['val_fn']
        self.test_output = result['test_output']
        self.input = result['input']
        self.mlp = result['mlp']

        self.image_width = None
        self.image_height = None
        self.image_path = pattern_config.IMAGE_FILE

    def nn_output(self, x):
        """Output of network for a given input."""
        return (theano.function(
            [self.input], [self.test_output])([x]))[0][0]
        # return (theano.function(
        #     [self.input], [T.argmax(self.test_output, axis=1)])([x]))[0][0]

    def get_coordinates_array(self, N, interpolation_factor=None):
        if not interpolation_factor:
            return range(N)
        else:
            if interpolation_factor < 1.:
                raise Exception('Invalid interpolation factor')
            interpolation_step = 1. / interpolation_factor
            # Ensure centering of image
            # Not if interpolation_factor is even number
            # (only centered for odd numbers)
            start = -int(interpolation_factor / 2) * interpolation_step
            end = N + start
            return np.arange(start, end, interpolation_step).tolist()

    def create_X(self, interpolation_factor=None):
        """Creates array of features (coordinates)."""
        height = self.get_image_height()
        width = self.get_image_width()
        data = []
        for i in self.get_coordinates_array(height, interpolation_factor):
            for j in self.get_coordinates_array(width, interpolation_factor):
                if self.coordinate_system == CoordinateSystem.CARTESIAN:
                    x = float(j)
                    y = float(i)
                    # NOTE: Sometimes repeating the same input multiple times helps.
                    data.append(np.array([x, y]))
                elif self.coordinate_system == CoordinateSystem.CARTESIAN_DOUBLE:
                    x1 = float(j)
                    y1 = float(i)
                    x2 = float(width - 1 - j)
                    y2 = float(height - 1 - i)
                    data.append(np.array([x1, y1, x2, y2]))
                elif self.pattern_config.COORDINATE_SYSTEM == CoordinateSystem.POLAR:
                    x = float(j) / width
                    y = float(i) / height
                    diff_from_center = (x - 0.5, y - 0.5)
                    r = math.sqrt(diff_from_center[0]**2 + diff_from_center[1]**2)
                    a = math.atan2(*diff_from_center)
                    data.append(np.array([r, a]))
                elif self.pattern_config.COORDINATE_SYSTEM == CoordinateSystem.DIST_FROM_CENTER:
                    center = (float(width - 1) / 2.,
                              float(height - 1) / 2.)
                    dist_x = abs(j - center[0])
                    dist_y = abs(i - center[1])
                    data.append(np.array([dist_x, dist_y]))
                elif self.coordinate_system == CoordinateSystem.X_CENTERED:
                    # Example Coordinates:
                    # TOP_LEFT: (1,0)
                    # TOP_RIGHT: (1,0)
                    # CENTER: (0,0.5)
                    # BOTTOM_LEFT: (1,1)
                    # BOTTOM_RIGHT: (1,1)

                    # To have coordinates centered need width to be odd.
                    # It's possible to enhance to support even widths.
                    assert width % 2 == 1
                    center = (float(width - 1) / 2.,
                              float(height - 1) / 2.)
                    dist_x = abs(j - center[0])
                    y = float(i)
                    # Normalize to [0,1] range
                    dist_x = float(dist_x) / math.floor(width / 2)
                    y = float(y) / (height - 1)
                    data.append(np.array([dist_x, y]))
                else:
                    raise Exception('Unk coordinate encoding.')
        return data
    
    def load_dataset(self):
        original = load_image(self.image_path)
        grayscale = convert_to_grayscale(original)

        data = self.create_X()
        labels = []
        height = self.get_image_height()
        width = self.get_image_width()
        for i in range(height):
            for j in range(width):
                if self.color_system == ColorSystem.BLACK_AND_WHITE:
                    if grayscale[i][j] == 1.:
                        labels.append(1)
                    elif grayscale[i][j] == 0.:
                        labels.append(0)
                    else:
                        raise Exception('Images should be black and white!')
                elif self.color_system == ColorSystem.GRAYSCALE:
                    labels.append(grayscale[i][j])
                else:
                    raise Exception('UNK color system {}'.format(
                        self.color_system))

        train_X = np.array(data, np.float32)

        if self.pattern_config.COORDINATE_SYSTEM == CoordinateSystem.X_CENTERED:
            # Making sure that points are symmetric in a case of X_CENTERED
            # One doesn't have to enforce this condition
            # if it wants to accept using it for pictures that are close to it.
            d = dict()
            for i in range(train_X.shape[0]):
                key = str(train_X[i][0]) + ':' + str(train_X[i][1])
                if key in d:
                    val = d[key]
                    assert val == labels[i]
                else:
                    d[key] = labels[i]

        # Scaler makes a big difference.
        self.scaler = StandardScaler()
        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X)

        train_Y = np.array(labels)
        if self.color_system == ColorSystem.BLACK_AND_WHITE:
            train_Y = np.float32(np.eye(self.outputs)[train_Y])
            # for hinge loss
            # make y be in {-1, 1}.
            train_Y = 2 * train_Y - 1.
        else:
            train_Y = train_Y.reshape((len(train_Y), 1))

        return train_X, train_Y

    def reconstruct_image(self, width, height, out_file,
                          interpolation_factor=None):
        data = self.create_X(interpolation_factor=interpolation_factor)
        data = np.array(data, np.float32)
        data = self.scaler.transform(data)
        labels = theano.function([self.input],
                                 [T.argmax(self.test_output, axis=1)])(data)[0]
        grayscale = []
        for i in range(len(labels)):
            grayscale.append(float(labels[i]))
        grayscale = np.array(grayscale)
        if interpolation_factor:
            grayscale = np.reshape(grayscale, (height * interpolation_factor,
                                               width * interpolation_factor))
        else:
            grayscale = np.reshape(grayscale, (height, width))
        rgb = (gray2rgb(grayscale) * 255).astype(np.uint8)
        cv2.imwrite(out_file, rgb)

    def translate(self, train_X, train_Y, shift_inputs=False):
        if shift_inputs:
            # Calculating the shift which if added to any training example
            # will make the example positive.
            inputs_shift = -np.min(train_X, axis=0)
            # adding epsilon
            inputs_shift += 0.001
        else:
            inputs_shift = None

        print_weight_counts(self.mlp)
        final_loss, final_err = self.val_fn(train_X, train_Y)
        print('final loss: ' + str(final_loss))
        print('final error: ' + str(final_err * 100) + '%')

        self.reconstruct_image(self.get_image_width(),
                               self.get_image_height(),
                               self.pattern_config.RECONSTRUCTED_IMAGE_FILE.format('final'))

        # Use all examples.
        examples_X = train_X
        examples_Y = train_Y

        # TODO: Change constant 'heart' when you change how models are saved
        # Reason is that pattern_config is saved in model pkl file,
        # and thus extending the structure to contain NAME is not trivial.
        results_dir = os.path.join(SIMULATION_RES_DIR, 'heart')

        with open(self.pattern_config.MATHEMATICA_FILE, 'w') as f:
            translate_to_crn(self.mlp, self.pattern_config.CRN_NAME,
                             f, inputs_shift)
            translate_examples(self.pattern_config.CRN_NAME,
                               examples_X,
                               examples_Y,
                               self.nn_output,
                               last_layer_num=len(self.hyperparams.HIDDEN_LAYERS_DIMS),
                               f=f,
                               tmax=50,
                               results_dir=results_dir,
                               inputs_shift=inputs_shift)
            translate_examples(self.pattern_config.CRN_NAME + 'Reduced',
                               examples_X,
                               examples_Y,
                               self.nn_output,
                               last_layer_num=len(self.hyperparams.HIDDEN_LAYERS_DIMS),
                               f=f,
                               tmax=50,
                               results_dir=results_dir,
                               inputs_shift=inputs_shift)

    def visualize(self, train_X, train_Y):
        layers = lasagne.layers.get_all_layers(self.mlp)
        hidden_layer = layers[2]
        W = hidden_layer.Wb.eval()
        b = hidden_layer.b.eval()

        neuron_values = np.matmul(train_X, W) + b
        neuron_values = np.maximum(neuron_values, 0.)

        # Test code
        # 3 hidden neurons in current model are not used at all.
        neuron_values2 = np.matmul(train_X, W[:,:5]) + b[:5]
        neuron_values2 = np.maximum(neuron_values2, 0.)
        out_layer = layers[4]
        W_out = out_layer.Wb.eval()
        b_out = out_layer.b.eval()
        out1 = np.matmul(neuron_values, W_out) + b_out
        out2 = np.matmul(neuron_values2, W_out[:5, :]) + b_out
        assert np.array_equal(out1, out2)

        inputs_increased = train_X + 2.
        b_transformed = np.copy(b)
        b_transformed[1] -= 2.
        b_transformed[2] -= 4.
        b_transformed[4] += 4.
        neuron_values3 = np.matmul(inputs_increased, W) + b_transformed
        neuron_values3 = np.maximum(neuron_values3, 0.)
        assert np.allclose(neuron_values[:,:5], neuron_values3[:,:5])

        for neuron_id in range(neuron_values.shape[1]):
            values = neuron_values[:,neuron_id]
            values = values.reshape(values.shape[0], 1)
            values = values.reshape(self.get_image_height(),
                                    self.get_image_width())

            c = plt.imshow(values, cmap=plt.get_cmap('Greys'))

            # draw gridlines
            plt.grid(which='major',
                     axis='both',
                     linestyle='-',
                     color='k',
                     linewidth=2)
            plt.xticks(np.arange(-.5, 17, 1))
            plt.yticks(np.arange(-.5, 15, 1))

            plt.colorbar(c, fraction=0.1)

            plt.tick_params(
                axis='both',  # changes apply to both axis.
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the left edge are off
                labelbottom=False,  # labels along the bottom edge are off
                labelleft=False  # labels along the left edge are off
            )

            plt.tight_layout()
            # plt.show()
            plt.savefig('/home/vasic/Downloads/neurons/neuron{}.pdf'
                        .format(neuron_id))
            plt.close()


    def callback_after_epoch(self, epoch_id):
        if epoch_id % 50 == 0:
            epoch_str = 'e{}'.format(epoch_id)
            file_name = self.pattern_config.RECONSTRUCTED_IMAGE_FILE.format(
                epoch_str
            )
            self.reconstruct_image(self.get_image_width(), self.get_image_height(),
                                   file_name)

    def get_image_width(self):
        if not self.image_width:
            original = load_image(self.image_path)
            self.image_width = original.shape[1]
        return self.image_width

    def get_image_height(self):
        if not self.image_height:
            original = load_image(self.image_path)
            self.image_height = original.shape[0]
        return self.image_height
    
    def train(self, save_files):
        train_X, train_Y = self.load_dataset()

        print('Training...')

        lr_decay = compute_LR_decay(
            LR_start=self.hyperparams.LR_start,
            LR_fin=self.hyperparams.LR_fin,
            NUM_EPOCHS=self.hyperparams.NUM_EPOCHS,
            LR_decay_type=self.hyperparams.LR_decay_type)

        callback_after_epoch = self.callback_after_epoch if save_files else None

        results = binary_connect.train(
            self.train_fn,
            self.val_fn,
            self.hyperparams.BATCH_SIZE,
            self.hyperparams.LR_start, lr_decay,
            self.hyperparams.NUM_EPOCHS,
            X_train=train_X, y_train=train_Y,
            X_val=train_X, y_val=train_Y,
            X_test=train_X, y_test=train_Y,
            network=self.mlp,
            return_best_epoch=True,
            callback_after_epoch=callback_after_epoch)

        save_model(self, self.pattern_config.MODEL_FILE)
        if save_files:
            self.reconstruct_image(
                self.get_image_width(),
                self.get_image_height(),
                self.pattern_config.RECONSTRUCTED_IMAGE_FILE.format('final'))

        print("Finished Training")

        # plt.plot(np.arange(len(training_losses)), training_losses)
        # plt.xlabel('epoch')
        # plt.ylabel('loss (train set)')
        # plt.show()

        return results

    def lr_rate_range_test(self):
        train_X, train_Y = self.load_dataset()

        print('Training...')

        start_lr = 1e-7
        end_lr = 0.5
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
            LR_start=start_lr, LR_decay=decay,
            num_epochs=num_epochs,
            X_train=train_X, y_train=train_Y,
            X_val=train_X, y_val=train_Y,
            X_test=train_X, y_test=train_Y,
            network=self.mlp,
            return_best_epoch=True,
            callback_after_epoch=None)

        print("Finished Training")

        plt.plot(learning_rates, result['training_losses'])
        plt.xlabel('LR')
        plt.xscale('log')
        plt.ylabel('loss (train set)')
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='HEART')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--range_test', action='store_true')
    parser.add_argument('--translate', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_files', action='store_true')
    parser.add_argument('--reconstruct_image', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pattern_config, hyperparams = get_pattern_configs(Pattern[args.pattern])
    if args.range_test:
        Model(hyperparams, pattern_config).lr_rate_range_test()
    if args.train:
        Model(hyperparams, pattern_config).train(args.save_files)
    if args.translate:
        model = load_model(pattern_config.MODEL_FILE)
        model.translate(*model.load_dataset())
    if args.visualize:
        model = load_model(pattern_config.MODEL_FILE)
        model.visualize(*model.load_dataset())
    if args.reconstruct_image:
        model = load_model(pattern_config.MODEL_FILE)
        model.reconstruct_image(model.get_image_width(),
                                model.get_image_height(),
                                model.pattern_config.RECONSTRUCTED_IMAGE_FILE.format('final'),
                                interpolation_factor=None)

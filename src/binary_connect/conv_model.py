# UNDER WORK
# NOT FINISHED YET

import re
from collections import OrderedDict
from collections import namedtuple
import theano
import theano.tensor as T
import lasagne

import binary_connect
import batch_norm


def create_conv_layer(net,
                      C, F, P,
                      binary,
                      stochastic,
                      H,
                      W_LR_scale,
                      nonlinearity):
    return binary_connect.Conv2DLayer(
        net,
        binary=binary,
        stochastic=stochastic,
        H=H,
        W_LR_scale=W_LR_scale,
        num_filters=C,
        filter_size=(F, F),
        pad=P,
        nonlinearity=nonlinearity)


def create_maxpool_layer(net, F):
    return lasagne.layers.MaxPool2DLayer(net, pool_size=(F, F))


def create_mlp_layer(net,
                     binary,
                     ternary,
                     zero_threshold,
                     stochastic,
                     H,
                     nonlinearity,
                     num_units):
    return binary_connect.DenseLayer(
        net,
        binary=binary,
        ternary=ternary,
        zero_threshold=zero_threshold,
        stochastic=stochastic,
        H=H,
        nonlinearity=nonlinearity,
        num_units=num_units)


def create_batch_norm_layer(net, epsilon, alpha, nonlinearity):
    return batch_norm.BatchNormLayer(
        net,
        epsilon=epsilon,
        alpha=alpha,
        nonlinearity=nonlinearity)


def create_dropout_layer(net, P):
    return lasagne.layers.DropoutLayer(net, P)


def make_model(hyperparams, input_shape):
    """

    :param hyperparams:
       Contains specification for the networks, with fields such as:
         layers
           A list of strings specifying layers of a neuraal network.
           'CxCONV-F-P' represents a CONV layer with C channels, FxF filters,
             and padding P.
           'MAXPOOL-F' represents a MAX-POOLING layers with FxF filter.
           'MLP-U' represents a feed-forward layer with U units.
           'BATCHNORM' represents a batch norm layer.
           'DROPOUT-P' reresents a dropout layer with probability P.
    :return:
    """

    input = T.tensor4('inputs')
    target = T.matrix('targets')

    net = lasagne.layers.InputLayer(
        shape=(None,) + input_shape,
        input_var=input)

    for i in range(len(hyperparams.LAYERS)):
        layer = hyperparams.LAYERS[i]

        m = re.match('(?P<C>.*)xCONV-(?P<F>.*)-(?P<P>.*)', layer)
        if m:
            C = int(m.group('C'))
            F = int(m.group('F'))
            P = int(m.group('P'))
            binary = (hyperparams.BINARIZATION == 'BINARY'
                      or hyperparams.BINARIZATION == 'TERNARY')
            stochastic = hyperparams.STOCHASTIC
            H = hyperparams.H
            nonlinearity = hyperparams.NONLINEARITIES[i]
            W_LR_scale = hyperparams.W_LR_scale

            net = create_conv_layer(net,
                                    C, F, P,
                                    binary=binary,
                                    stochastic=stochastic,
                                    H=H,
                                    nonlinearity=nonlinearity,
                                    W_LR_scale=W_LR_scale)
            continue

        m = re.match('MAXPOOL-(?P<F>.*)', layer)
        if m:
            F = int(m.group('F'))
            net = create_maxpool_layer(net, F)
            continue

        m = re.match('MLP-(?P<U>.*)', layer)
        if m:
            U = int(m.group('U'))
            binary = hyperparams.BINARIZATION == 'BINARY'
            ternary = hyperparams.BINARIZATION == 'TERNARY'
            zero_threshold = hyperparams.ZERO_THRESHOLD
            stochastic = hyperparams.STOCHASTIC
            H = hyperparams.H
            nonlinearity = hyperparams.NONLINEARITIES[i]
            net = create_mlp_layer(net,
                                   binary=binary,
                                   ternary=ternary,
                                   zero_threshold=zero_threshold,
                                   stochastic=stochastic,
                                   H=H,
                                   nonlinearity=nonlinearity,
                                   num_units=U)
            continue

        m = re.match('BATCHNORM', layer)
        if m:
            epsilon = hyperparams.EPSILON,
            alpha = hyperparams.ALPHA,
            nonlinearity = lasagne.nonlinearities.identity
            net = create_batchnorm_layer(net,
                                         epsilon=epsilon,
                                         alpha=alpha,
                                         nonlinearity=nonlinearity)
            continue

        m = re.match('DROPOUT-(?P<P>.*)', layer)
        if m:
            P = float(m.group('P'))
            net = create_dropout_layer(net, P)
            continue

        raise Exception('Unrecognized layer type: {}'.format(layer))

    train_output = lasagne.layers.get_output(net, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))

    if hyperparams.BINARIZATION:

        # W updates
        W = lasagne.layers.get_all_params(net, binary=True)
        W_grads = binary_connect.compute_grads(loss, net)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W,
                                       learning_rate=LR)
        updates = binary_connect.clipping_scaling(updates, net)

        # other parameters updates
        params = lasagne.layers.get_all_params(net, trainable=True,
                                               binary=False)
        updates = OrderedDict(
            updates.items() + lasagne.updates.adam(loss_or_grads=loss,
                                                   params=params,
                                                   learning_rate=LR).items())

    else:
        params = lasagne.layers.get_all_params(net, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params,
                                       learning_rate=LR)

    test_output = lasagne.layers.get_output(net, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
    test_err = T.mean(
        T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),
        dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    result = dict()
    result['train_fn'] = train_fn
    result['val_fn'] = val_fn
    result['test_output'] = test_output
    result['input'] = input
    result['net'] = net
    return result


HyperparamsConfig = namedtuple('HyperparamsConfig', [
    'BATCH_SIZE',
    # alpha is the exponential moving average factor
    'ALPHA',
    'EPSILON',
    'LAYERS',
    'NONLINEARITIES',
    'NUM_EPOCHS',
    # Can be BINARY, TERNARY, or None for no binarization.
    'BINARIZATION',
    'ZERO_THRESHOLD',
    'STOCHASTIC',
    # (-H,+H) are the two binary values
    # H = "Glorot"
    'H',
    # "Glorot" means we are using the coefficients from Glorot's paper
    'W_LR_scale',
    'LR_start',
    'LR_fin',
    'LR_decay_type'
    ])

hyperparams = HyperparamsConfig(
    BATCH_SIZE=100,
    ALPHA=.15,
    EPSILON=1e-4,
    LAYERS=['32xCONV-3-0', 'DROPOUT-0.5', '32xCONV-3-0', 'MLP-64'],
    NONLINEARITIES=[lasagne.nonlinearities.rectify,
                    None,
                    lasagne.nonlinearities.rectify,
                    lasagne.nonlinearities.rectify],
    NUM_EPOCHS=250,
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


if __name__ == '__main__':
    net = make_model(hyperparams, input_shape=(3, 32, 32))
    pass

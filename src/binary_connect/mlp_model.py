import theano
import theano.tensor as T
import lasagne
from collections import OrderedDict

import binary_connect
import batch_norm


def make_model(hyperparams,
               INPUT_DIMS,
               OUTPUT_DIMS,
               loss_func='HINGE',
               make_dropout_layer_when_zero=False):
    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    if loss_func == 'HINGE':
        target = T.matrix('targets')
    elif loss_func == 'MSE':
        target = T.matrix('targets')

    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
        shape=(None, INPUT_DIMS),
        input_var=input)

    if make_dropout_layer_when_zero or hyperparams.DROPOUT_IN != 0.:
        mlp = lasagne.layers.DropoutLayer(
            mlp,
            p=hyperparams.DROPOUT_IN)

    for k in range(len(hyperparams.HIDDEN_LAYERS_DIMS)):
        mlp = binary_connect.DenseLayer(
            mlp,
            binary=hyperparams.BINARIZATION == 'BINARY',
            ternary=hyperparams.BINARIZATION == 'TERNARY',
            zero_threshold=hyperparams.ZERO_THRESHOLD,
            stochastic=hyperparams.STOCHASTIC,
            H=hyperparams.H,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=hyperparams.HIDDEN_LAYERS_DIMS[k])

        if hyperparams.USE_BATCH_NORMALIZATION:
            mlp = batch_norm.BatchNormLayer(
                mlp,
                epsilon=hyperparams.EPSILON,
                alpha=hyperparams.ALPHA,
                nonlinearity=lasagne.nonlinearities.identity)

        mlp.nonlinearity = lasagne.nonlinearities.rectify

        if make_dropout_layer_when_zero or hyperparams.DROPOUT_HIDDEN != 0.:
            mlp = lasagne.layers.DropoutLayer(
                mlp,
                p=hyperparams.DROPOUT_HIDDEN)

    mlp = binary_connect.DenseLayer(
        mlp,
        binary=hyperparams.BINARIZATION == 'BINARY',
        ternary=hyperparams.BINARIZATION == 'TERNARY',
        zero_threshold=hyperparams.ZERO_THRESHOLD,
        stochastic=hyperparams.STOCHASTIC,
        H=hyperparams.H,
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=OUTPUT_DIMS)

    if hyperparams.USE_BATCH_NORMALIZATION:
        mlp = batch_norm.BatchNormLayer(
            mlp,
            epsilon=hyperparams.EPSILON,
            alpha=hyperparams.ALPHA,
            nonlinearity=lasagne.nonlinearities.identity)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)

    if loss_func == 'HINGE':
        # squared hinge loss
        loss = T.mean(T.sqr(T.maximum(0., 1. - target * train_output)))
    elif loss_func == 'MSE':
        # mean squared error
        loss = T.mean(T.sqr(target - train_output))
    else:
        raise Exception("UNK loss function {}".format(loss_func))

    if hyperparams.BINARIZATION:
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_connect.compute_grads(loss, mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W,
                                       learning_rate=LR)
        updates = binary_connect.clipping_scaling(updates, mlp)

        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True,
                                               binary=False)
        updates = OrderedDict(
            updates.items() + lasagne.updates.adam(loss_or_grads=loss,
                                                   params=params,
                                                   learning_rate=LR).items())

    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params,
                                       learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)

    if loss_func == 'HINGE':
        test_loss = T.mean(T.sqr(T.maximum(0., 1. - target * test_output)))
        test_err = T.mean(T.neq(T.argmax(test_output, axis=1),
                                T.argmax(target, axis=1)),
                          dtype=theano.config.floatX)
    elif loss_func == 'MSE':
        test_loss = T.mean(T.sqr(target - test_output))
        test_err = test_loss
    else:
        raise Exception("UNK loss function {}".format(loss_func))

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
    result['mlp'] = mlp
    return result
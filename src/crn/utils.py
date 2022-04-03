import lasagne
import numpy as np
import pickle
import sys

from binary_connect import batch_norm
from binary_connect import binary_connect

# Name of the input species.
INPUT_S_NAME = 'x{idx}S{sign}'
# Name of the layer output species.
LAYER_S_NAME = 'hL{layer_idx}N{neuron_idx}S{sign}'
# Name of the layer intermediate species.
LAYER_INTER_S_NAME = 'iL{layer_idx}N{neuron_idx}S{sign}'

# Postfix for positive species.
POS_SPECIES = 'pos'
# Postfix for negative species.
NEG_SPECIES = 'neg'


def save_model(model, model_file):
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(old_limit * 5)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    sys.setrecursionlimit(old_limit)
    # np.savez(MODEL_FILE, lasagne.layers.get_all_param_values(
    #     self.mlp))


def load_model(model_file):
    with open(model_file, 'rb') as f:
        return pickle.load(f)
    # npz = np.load(MODEL_FILE)
    # lasagne.layers.set_all_param_values(self.mlp, npz)


def simulate_network(network, inputs):
    layers = lasagne.layers.get_all_layers(network)

    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, lasagne.layers.input.InputLayer):
            mlp = inputs
        elif isinstance(layer, lasagne.layers.noise.DropoutLayer):
            continue
        elif isinstance(layer, binary_connect.DenseLayer):
            W = layer.Wb.eval()
            b = layer.b.eval()
            mlp = np.dot(mlp, W) + b
        elif isinstance(layer, batch_norm.BatchNormLayer):
            batch_info = {
                'mean': layer.mean.eval(),
                'std': layer.std.eval(),
                'beta': layer.beta.eval(),
                'gamma': layer.gamma.eval()
            }
            # Reduce to 0-mean, unit variance
            mlp = (mlp - batch_info['mean']) / batch_info['std']
            # Add Batch Norm parameters
            mlp = (mlp * batch_info['gamma']) + batch_info['beta']
        else:
            raise Exception('Unknown layer type: ' + str(type(layers)))

        if hasattr(layer, 'nonlinearity'):
            nonlinearity = layer.nonlinearity.func_name
            if nonlinearity == 'linear':
                pass
            elif nonlinearity == 'rectify':
                mlp = np.maximum(mlp, 0.)
            else:
                raise Exception(
                    'Unsupported nonlinearity: ' + str(nonlinearity))
    print("Network simulated model output:")
    print(mlp)


def squash_batch_normalization(W, b, batch_norm_layer):
    """
    Squashes batch norm layer into weights and biases, i.e. it returns modified
    W (W1) and b (b1) such that the transformation is equivalent:
    batch_norm(X * W + b) == X * W1 + b1

    It modifies W and b in place.
    """
    batch_info = {
        'mean': batch_norm_layer.mean.eval(),
        'std':  batch_norm_layer.std.eval(),
        'beta': batch_norm_layer.beta.eval(),
        'gamma': batch_norm_layer.gamma.eval()
    }
    for row_id in range(W.shape[0]):
        for column_id in range(W.shape[1]):
            factor = batch_info['gamma'][0][column_id] / batch_info['std'][0][
                column_id]
            W[row_id][column_id] *= factor
            if row_id == 0:
                b[column_id] = (b[column_id] - batch_info['mean'][0][
                    column_id]) * factor + batch_info['beta'][0][column_id]


def simulate_network_squash_batching(network, inputs):
    """Assumes that a BatchLayer always comes after DenseLayer"""
    layers = lasagne.layers.get_all_layers(network)

    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, lasagne.layers.input.InputLayer):
            mlp = inputs
        elif isinstance(layer, lasagne.layers.noise.DropoutLayer):
            continue
        elif isinstance(layer, binary_connect.DenseLayer):
            W = layer.Wb.eval()
            b = layer.b.eval()

            if (i < len(layers) - 1 and
                    isinstance(layers[i + 1], batch_norm.BatchNormLayer)):
                squash_batch_normalization(W, b, layers[i + 1])

            mlp = np.dot(mlp, W) + b
        elif isinstance(layer, batch_norm.BatchNormLayer):
            pass
        else:
            raise Exception('Unknown layer type: ' + str(type(layers)))

        if hasattr(layer, 'nonlinearity'):
            nonlinearity = layer.nonlinearity.func_name
            if nonlinearity == 'linear':
                pass
            elif nonlinearity == 'rectify':
                mlp = np.maximum(mlp, 0.)
            else:
                raise Exception(
                    'Unsupported nonlinearity: ' + str(nonlinearity))
    print("Network simulated (batching squashed) model output:")
    print(mlp)


def approximate_batch_layers(network):
    layers = lasagne.layers.get_all_layers(network)
    for i in range(len(layers)):
        layer = layers[i]
        if (isinstance(layer, binary_connect.DenseLayer)
            and i < len(layers) - 1
            and isinstance(layers[i + 1], batch_norm.BatchNormLayer)):
            Wb = layer.Wb.eval()
            W = layer.Wb.eval()
            b = layer.b.eval()
            Wb_old = np.copy(W)
            W_old = np.copy(W)
            b_old = np.copy(b)
            squash_batch_normalization(W, b, layers[i + 1])
            layer.W.set_value(np.clip(np.rint(W), a_min=-1, a_max=1))
            layer.b.set_value(b)
        if (isinstance(layer, batch_norm.BatchNormLayer)):
            # Resets the effects of BatchNormLayer.
            mean = layer.mean.eval()
            layer.mean.set_value(np.zeros(shape=mean.shape, dtype=mean.dtype))
            std = layer.std.eval()
            layer.std.set_value(np.ones(shape=std.shape, dtype=std.dtype))
            gamma = layer.gamma.eval()
            layer.gamma.set_value(np.ones(shape=gamma.shape, dtype=gamma.dtype))
            beta = layer.beta.eval()
            layer.beta.set_value(np.zeros(shape=beta.shape, dtype=beta.dtype))


def get_input_names(num_inputs, postfix, separator=','):
    s = ''
    for input_id in range(num_inputs):
        s += INPUT_S_NAME.format(idx=input_id, sign=POS_SPECIES)
        s += postfix
        s += separator
        s += INPUT_S_NAME.format(idx=input_id, sign=NEG_SPECIES)
        s += postfix
        if input_id < num_inputs - 1:
            s += separator
    return s


def translate_to_crn(network, name, f, inputs_shift=None):
    layers = lasagne.layers.get_all_layers(network)

    f.write('#!/usr/bin/env wolframscript\n\n')
    f.write('Get[Directory[] <> "/CRNSimulator.m"];\n')
    f.write('Get[Directory[] <> "/utils.m"];\n\n')

    signs = [POS_SPECIES, NEG_SPECIES]

    # Assumes that first dimension is None, and second equal to input dims.
    input_dims = layers[0].shape[1]

    # FUNCTION HEADER #
    f.write(name + '[')
    f.write(get_input_names(input_dims, postfix='Initial_', separator=','))
    f.write('] :=\n')
    f.write('{\n')

    # SETTING INITIAL INPUT SPECIES CONCENTRATIONS #
    for input_id in range(input_dims):
        for sign_id in range(len(signs)):
            sign = signs[sign_id]

            if inputs_shift is not None and sign == NEG_SPECIES:
                continue

            s_name = INPUT_S_NAME.format(
                idx=input_id,
                sign=sign)
            s_name_initial = s_name + 'Initial'
            spacing = ',\n' if input_id > 0 or sign_id > 0 else ''
            f.write('{}conc[{}, {}]'.format(spacing, s_name, s_name_initial))

    layer_num = -1

    for layer_id in range(len(layers)):
        layer = layers[layer_id]
        if isinstance(layer, lasagne.layers.input.InputLayer):
            continue
        elif isinstance(layer, lasagne.layers.noise.DropoutLayer):
            continue
        elif isinstance(layer, binary_connect.DenseLayer):
            layer_num += 1
            W = layer.Wb.eval()
            b = layer.b.eval()
            assert W.shape[1] == b.shape[0]

            relu_activation = False
            if hasattr(layer, 'nonlinearity'):
                activation = layer.nonlinearity.func_name
                if activation == 'rectify':
                    relu_activation = True
                if activation != 'linear' and activation != 'rectify':
                    raise Exception('Unsupported nonlinearity: '
                                    + str(nonlinearity))

            for row_id in range(W.shape[0]):
                for sign_id in range(len(signs)):
                    sign = signs[sign_id]
                    opposite_sign = signs[(sign_id + 1) % len(signs)]

                    if (inputs_shift is not None
                        and layer_num == 0
                        and sign == NEG_SPECIES):
                        continue

                    rxn_string = ',\nrxn['
                    reactant_species = (
                        INPUT_S_NAME.format(idx=row_id, sign=sign)
                        if layer_num == 0
                        else LAYER_S_NAME.format(layer_idx=layer_num - 1,
                                                 neuron_idx=row_id,
                                                 sign=sign))
                    rxn_string += reactant_species
                    rxn_string += ','
                    # True if all products are empty
                    products_empty = True
                    for column_id in range(W.shape[1]):
                        if W[row_id][column_id] == 1.:
                            product_sign = sign
                        elif W[row_id][column_id] == -1.:
                            product_sign = opposite_sign
                        elif W[row_id][column_id] == 0.:
                            continue
                        else:
                            raise Exception('Unsupported weight')
                        if not products_empty:
                            rxn_string += '+'
                        if (W[row_id][column_id] == 1.
                                or W[row_id][column_id] == -1.):
                            products_empty = False

                        # If no Relu activation, immediately output layer
                        # output species.
                        species_name = (
                            LAYER_INTER_S_NAME
                            if relu_activation else LAYER_S_NAME
                        )
                        rxn_string += species_name.format(
                            layer_idx=layer_num,
                            neuron_idx=column_id,
                            sign=product_sign)
                    if not products_empty:
                        rxn_string += ',1]'
                        f.write(rxn_string)

            for column_id in range(b.shape[0]):
                number = b[column_id]
                if type(b[column_id]).__name__ == 'GpuArray':
                    number = float(str(b[column_id]))

                if inputs_shift is not None and layer_num == 0:
                    for row_id in range(W.shape[0]):
                        number -= inputs_shift[row_id] * W[row_id][column_id]

                if number >= 0.:
                    species_name = (LAYER_INTER_S_NAME if relu_activation
                        else LAYER_S_NAME)
                    s_name = species_name.format(
                        layer_idx=layer_num,
                        neuron_idx=column_id,
                        sign=POS_SPECIES)
                    f.write(',\nconc[{},{:.10f}]'.format(
                        s_name,
                        number
                    ))
                else:
                    species_name = (LAYER_INTER_S_NAME if relu_activation
                        else LAYER_S_NAME)
                    s_name = species_name.format(
                        layer_idx=layer_num,
                        neuron_idx=column_id,
                        sign=NEG_SPECIES)
                    f.write(',\nconc[{},{:.10f}]'.format(
                        s_name,
                        -number
                    ))

            if relu_activation:
                W = layer.Wb.eval()
                for i in range(W.shape[1]):
                    x_p = LAYER_INTER_S_NAME.format(
                        layer_idx=layer_num,
                        neuron_idx=i,
                        sign=POS_SPECIES)
                    x_n = LAYER_INTER_S_NAME.format(
                        layer_idx=layer_num,
                        neuron_idx=i,
                        sign=NEG_SPECIES)
                    m_species = 'mL{}N{}'.format(layer_num, i)
                    y_p = LAYER_S_NAME.format(
                        layer_idx=layer_num,
                        neuron_idx=i,
                        sign=POS_SPECIES)
                    y_n = LAYER_S_NAME.format(
                        layer_idx=layer_num,
                        neuron_idx=i,
                        sign=NEG_SPECIES)
                    f.write(',\nrxn[{},{}+{},1]'.format(
                        x_p, m_species, y_p
                    ))
                    f.write(',\nrxn[{}+{},{},1]'.format(
                        m_species, x_n, y_n
                    ))
        elif isinstance(layer, batch_norm.BatchNormLayer):
            raise Exception('Unsupported translation of BatchNormLayer.')
        else:
            raise Exception('Unknown layer type: ' + str(type(layers)))

    f.write('\n}\n\n')

    # REDUCED CRN FUNCTION HEADER #
    f.write(name + 'Reduced' + '[')

    f.write(get_input_names(input_dims, postfix='Initial_', separator=','))
    f.write('] =\n')

    f.write('  reduceFFNCCrn[' + name + '[')
    f.write(get_input_names(input_dims, postfix='Initial', separator=','))
    f.write('], keepInputLayer=True, inputPrefix="x"];\n\n')

    f.write('cnt = Count[' + name + '[' + get_input_names(input_dims, postfix='Initial') + '], rxn[___]];\n')
    f.write('Print["Number of reactions: " <> ToString[cnt]];\n')
    f.write('cntReduced = Count[' + name + 'Reduced[' + get_input_names(input_dims,
                                                      postfix='Initial') + '], rxn[___]];\n')
    f.write('Print["Number of reactions (reduced form): " <> ToString[cntReduced]];\n')
    f.write('\n')


def print_outputs(output_dims, last_layer_num, time_string, f):
    """
    Creates a Mathematica list of subtracted output species.
    e.g.:
    {(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol};
    would be written to file f for
    output_dims=2, last_layer_num=1, time_string='t'
    """
    out_species_name = 'hL{layer_idx}N{neuron_idx}S{sign}'

    f.write('{')
    for out_idx in range(output_dims):
        f.write('(')
        f.write(out_species_name.format(
            layer_idx=last_layer_num,
            neuron_idx=out_idx,
            sign=POS_SPECIES) + '[' + time_string + '] - ')
        f.write(out_species_name.format(
            layer_idx=last_layer_num,
            neuron_idx=out_idx,
            sign=NEG_SPECIES) + '[' + time_string + ']')
        f.write(')')
        f.write('/.sol')
        if out_idx < output_dims - 1:
            f.write(',')
    f.write('};\n')


def print_outputs_summed(output_dims, last_layer_num, time_string, f):
    """
    Creates a Mathematica expression summing all output signals at the given
    time moment.
    e.g.:
    (hL1N0Spos[t] + hL1N0Sneg[t] + hL1N1Spos[t] + hL1N1Sneg[t])/.sol;
    would be written to file f for
    output_dims=2, last_layer_num=1, time_string='t'
    """
    out_species_name = 'hL{layer_idx}N{neuron_idx}S{sign}'

    f.write('{(')
    for out_idx in range(output_dims):
        f.write(out_species_name.format(
            layer_idx=last_layer_num,
            neuron_idx=out_idx,
            sign=POS_SPECIES) + '[' + time_string + ']')
        f.write(' + ')
        f.write(out_species_name.format(
            layer_idx=last_layer_num,
            neuron_idx=out_idx,
            sign=NEG_SPECIES) + '[' + time_string + ']')
        if out_idx < output_dims - 1:
            f.write(' + ')
    f.write(')/.sol};\n')


def print_outputs_raw(output_dims, last_layer_num, time_string, f):
    """
    Creates a Mathematica expression with all output species.
    e.g.:
    {(hL1N0Spos[t]) /. sol, (hL1N0Sneg[t]) /. sol, (hL1N1Spos[t]) /. sol, (hL1N1Sneg[t])/.sol}
    would be written to file f for
    output_dims=2, last_layer_num=1, time_string='t'
    """
    out_species_name = 'hL{layer_idx}N{neuron_idx}S{sign}'

    f.write('{')
    for out_idx in range(output_dims):
        f.write('(')
        f.write(out_species_name.format(
            layer_idx=last_layer_num,
            neuron_idx=out_idx,
            sign=POS_SPECIES) + '[' + time_string + ']')
        f.write(') /.sol ,(')
        f.write(out_species_name.format(
            layer_idx=last_layer_num,
            neuron_idx=out_idx,
            sign=NEG_SPECIES) + '[' + time_string + ']')
        f.write(')')
        f.write('/.sol')
        if out_idx < output_dims - 1:
            f.write(',')
    f.write('};\n')


def np_array_to_string(a):
    suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    result = np.array2string(a, precision=15, separator=',')
    np.set_printoptions(suppress=suppress)
    return result


def translate_examples(name,
                       X, y,
                       nn_output, last_layer_num, f,
                       results_dir,
                       tmax=50,
                       inputs_shift=None):
    """
    Translates dataset (x,y) examples into Mathematica code that performs
    simulations for those values.
    :param name: Name of the experiment.
    :param X: Numpy 2D array of shape (N, F); where N is the number of examples,
      and F number of features that characterize one example.
    :param y: Numpy 2D array of shape (N, C); where N is number of examples,
      and C is number of classes. It is a one-hot encoding format where allowed
      values are {1, -1}.
    :param nn_output: This is a function which given an input x, produces
      output values of neural network -- consists of a list with values
      of all output units.
    :param last_layer_num: Number of the last layer.
    :param tmax: Maximum simulation time used.
    :param f: File to write to.
    :param results_dir: Directory where simulation results will be saved.
    :param inputs_shift: This represents amount for which inputs
      should be shifted. It is used to eliminate need for dual rail inputs.
      E.g., if input x1 is in range [-1,1]. We can transform it to be in range
      [0,2]. However, that requires adjusting bias terms accordingly.
      If NONE, inputs are not shifted. If provided it should be of dimension
      (F, 1)
    """

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert inputs_shift is None or inputs_shift.shape == (X.shape[1],)

    f.write('tmax = {};\n'.format(tmax))
    f.write('totalCount = 0;\n')
    f.write('correctCount = 0;\n')
    f.write('matchingCount = 0;\n\n')
    f.write('analyzeConvergenceTime = False;\n\n')

    f.write('Print["Example Index (i)"];\n')
    f.write('Print["Expected Output (E)"];\n')
    f.write('Print["CRN Output (C)"];\n')
    f.write('Print["NN Output (N)"];\n\n')
    f.write('Print["i: E C N"];\n')

    for i in range(X.shape[0]):
        current_X = np.copy(X[i])
        if inputs_shift is not None:
            current_X += inputs_shift
            if np.min(current_X) < 0:
                raise Exception('Inputs should be shifted in a way to make them positive!')

        current_y = y[i]

        f.write('rsys=' + name + '[')
        for feat_idx in range(current_X.shape[0]):
            if current_X[feat_idx] > 0:
                f.write('{:.6f}'.format(current_X[feat_idx]) + ',')
                f.write('0')
                if feat_idx < current_X.shape[0] - 1:
                    f.write(',')
            else:
                f.write('0,')
                f.write('{:.6f}'.format(-current_X[feat_idx]))
                if feat_idx < current_X.shape[0] - 1:
                    f.write(',')
        f.write('];\n')

        f.write('sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];\n')

        f.write('results=')
        print_outputs(current_y.shape[0], last_layer_num, 'tmax', f)

        f.write('predicted = Position[results,Max[results]][[1]][[1]] - 1;\n')
        f.write('totalCount += 1;\n')
        f.write('warning = "";\n')
        f.write('correctOutput = {};\n'.format(np.argmax(current_y)))
        if inputs_shift is not None:
            current_nn_output = nn_output(current_X - inputs_shift)
        else:
            current_nn_output = nn_output(current_X)
        assert isinstance(current_nn_output, np.ndarray), "nn_output should produce output for all units of NN."
        assert current_nn_output.shape[0] == y.shape[1]
        f.write('nnOutput = {};\n'.format(np.argmax(current_nn_output)))
        f.write('If[predicted == correctOutput, correctCount+=1, warning = "*"];\n')
        f.write('If[predicted == nnOutput, matchingCount+=1, warning = "**"];\n')

        f.write(
            'Print["{}: {} " <> ToString[predicted] <> " {} " <> warning];\n'
                .format(i,
                        np.argmax(current_y),
                        np.argmax(current_nn_output)))

        f.write('correctResults=List{};\n'.format(
            np_array_to_string(current_nn_output)))

        f.write('plotter=')
        print_outputs(current_y.shape[0], last_layer_num, 't', f)

        f.write('p' + str(i) + '=Plot[plotter, {t, 0, tmax}, PlotLabels->{')
        for z in range(current_y.shape[0]):
            f.write('"y{}"'.format(z))
            if z < current_y.shape[0] - 1:
                f.write(',')
        f.write('}, PlotRange->All, AxesLabel->{"time","value"}];\n')
        f.write('Export[FileNameJoin[{{"{}", "{}_p{}.svg"}}], p{}];\n'.format(
            results_dir, name, i, i))

        f.write('\n')

        f.write('If[analyzeConvergenceTime, \n')
        f.write('  outputConfidences = softmax[correctResults];\n')
        f.write('  correctOutputConfidence = outputConfidences[[correctOutput+1]];\n')
        output_file_name = '{}_convergence_times.csv'.format(name)
        f.write('  WriteString["{}", ToString[DecimalForm[correctOutputConfidence,{{20,15}}]] <> ","];\n'
                .format(output_file_name))
        f.write('  WriteString["{}", ToString[DecimalForm[Max[outputConfidences],{{20,15}}]] <> ","];\n'
                .format(output_file_name))
        f.write('  times = AnalyzeConvergence[plotter, correctResults, correctOutput + 1, tmax];\n')
        f.write('  For[i=1, i <= Length[times], i = i+1,\n')
        f.write('    WriteString["{}", DecimalForm[times[[i]], {{20, 5}}]];\n'
                .format(output_file_name))
        f.write('    WriteString["{}", ","];\n'.format(output_file_name))
        f.write('  ];\n')

        f.write('  WriteString["{}", "{:6f},"];\n'
                .format(output_file_name, np.sum(np.abs(current_X))))

        f.write('  plotterSummed=')
        print_outputs_summed(current_y.shape[0], last_layer_num, 't', f)
        f.write('  summedOutputs = (plotterSummed /. {t->tmax})[[1]];\n')
        f.write('  WriteString["{}", DecimalForm[summedOutputs,{{20,15}}]];\n'
                .format(output_file_name))

        # f.write('  plotterRaw=')
        # print_outputs_raw(current_y.shape[0], last_layer_num, 't', f)
        # f.write('  convergenceTime=Mean[EpsilonPercentConvergenceTimes[plotterRaw, plotterRaw /. {t->tmax}, tmax, 0.9, tdelta = 0.05]];\n')
        # f.write('  WriteString["{}", DecimalForm[convergenceTime,{{20,15}}]];\n'
        #         .format(output_file_name))

        f.write('  WriteLine["{}", ""];\n'.format(output_file_name))

        f.write('];\n')

        f.write('\n')

    f.write('Print["totalCount (number of predictions): " <> ToString[totalCount]];\n')
    f.write('Print["correctCount (number of correct predictions): " <> ToString[correctCount]];\n')
    f.write('Print["matchingCount (number of predictions matching NN output): " <> ToString[matchingCount]];\n')


def translate_convergence_experiment(name,
                                     X,
                                     nn_output,
                                     num_output_units,
                                     last_layer_num,
                                     f,
                                     tmax,
                                     convergence_types):
    """

    """
    assert isinstance(X, np.ndarray)

    f.write('tmax={};\n\n'.format(tmax))

    for i in range(X.shape[0]):
        current_X = X[i]

        f.write('rsys=' + name + '[')
        for feat_idx in range(current_X.shape[0]):
            if current_X[feat_idx] > 0:
                f.write('{:.6f}'.format(current_X[feat_idx]) + ',')
                f.write('0')
                if feat_idx < current_X.shape[0] - 1:
                    f.write(',')
            else:
                f.write('0,')
                f.write('{:.6f}'.format(-current_X[feat_idx]))
                if feat_idx < current_X.shape[0] - 1:
                    f.write(',')
        f.write('];\n')

        f.write('sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];\n')

        current_nn_output = nn_output(current_X)
        assert isinstance(current_nn_output, np.ndarray), \
            "nn_output should produce output for all units of NN."
        assert current_nn_output.shape[0] == num_output_units

        f.write('correctResults=List{};\n\n'.format(
            np_array_to_string(current_nn_output)))

        output_file_name = '{}_convergence_times.csv'.format(name)

        summed_output_convergence = False
        convergence_string = None
        for convergence_type in convergence_types:
            split = convergence_type.split('-')
            convergence_name = str(split[0])
            if len(split) == 1:
                convergence_parameter = 'Null'
            else:
                convergence_parameter = str(split[1])

            if convergence_name == 'OutputsSummed':
                summed_output_convergence = True
                continue

            if not convergence_string:
                convergence_string = '{'
            else:
                convergence_string += ','
            convergence_string += '{{"{}",{}}}'.format(convergence_name,
                                                       convergence_parameter)
        if convergence_string:
            convergence_string += '}'

        if convergence_string:
            f.write('plotter=')
            print_outputs(num_output_units, last_layer_num, 't', f)

            # No need for plot. I'll just comment it out in Mathematica.
            f.write('(* ')
            f.write('p' + str(i) + '=Plot[plotter, {t, 0, tmax}, PlotLabels->{')
            for z in range(num_output_units):
                f.write('"y{}"'.format(z))
                if z < num_output_units - 1:
                    f.write(',')
            f.write('}, PlotRange->All, AxesLabel->{"time","value"}]; *) \n')

            f.write('\n')

            f.write('convergenceTimes = ComputeConvergenceMultiple[plotter, correctResults, tmax, {}, tdelta=0.1];\n'.format(convergence_string))
            f.write('For[i=1, i <= Length[convergenceTimes], i++,\n')
            f.write('  WriteString["{}", ToString[DecimalForm[convergenceTimes[[i]], {{20, 5}}]]];\n'.format(output_file_name))
            f.write('  If[i < Length[convergenceTimes], WriteString["{}", ","]];\n'.format(output_file_name))
            f.write('];\n')

        if summed_output_convergence:
            if convergence_string:
                f.write('WriteString["{}", ","];\n'.format(output_file_name))

            f.write('plotterSummed=')
            print_outputs_summed(num_output_units, last_layer_num, 't', f)
            # TODO:
            # Note that using plotterSummed /. {t->tmax} as the target value to aim for
            # is not a correct thing to do; the correct thing would be to compute
            # correct values and set them as targets.
            # For now, setting tmax to a high value would be sufficient.
            f.write('convergenceTimeSummedSignal = EpsilonPercentConvergenceTimes[plotterSummed, plotterSummed /. {t->tmax}, tmax, 0.9, tdelta=0.1][[1]];\n')
            f.write(
                'WriteString["{}", ToString[DecimalForm[convergenceTimeSummedSignal, {{20, 5}}]]];\n'.format(
                    output_file_name))

        f.write('WriteString["{}", "\\n"];\n\n'.format(output_file_name))

        f.write('\n')

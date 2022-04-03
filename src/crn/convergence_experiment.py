import os
import pathlib2
import stat
import subprocess
import shutil
import random
import string
import tempfile
import numpy as np
import pandas as pd
import theano
import lasagne
import itertools
import csv
import matplotlib.pyplot as plt
from collections import namedtuple
from datetime import datetime
from binary_connect.mlp_model import make_model
from crn.hyperparams import HyperparamsConfig
from crn.utils import translate_to_crn
from crn.utils import translate_convergence_experiment


ExperimentResult = namedtuple('ExperimentResult',
                               ['model_id',
                                'num_input_units',
                                'num_output_units',
                                'num_hidden_layers',
                                'num_hidden_units',
                                'convergence_time',
                                'inputs',
                                'parameters',
                                'date'])


class ConvergenceExperiment(object):
    def __init__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='CRN_convergence_')
        libs_dir = os.path.join(os.getcwd(), 'data', 'mathematica')
        shutil.copyfile(os.path.join(libs_dir, 'CRNSimulator.m'),
                        os.path.join(self.tmp_dir, 'CRNSimulator.m'))
        shutil.copyfile(os.path.join(libs_dir, 'utils.m'),
                        os.path.join(self.tmp_dir, 'utils.m'))

    def create_model_id(self):
        return ''.join(np.random.choice(list(string.ascii_uppercase),
                                        size=15).tolist())

    def generate_random_NN(self,
                           num_input_units,
                           num_hidden_layers,
                           num_hidden_units,
                           num_output_units,
                           bias_zero=True):
        """
        Creates a random neural network.
        Real-valued weights and bias terms are assigned to real values in range
        [-1,1] uniformly sampled.
        Note that the real-valued weights are afterward binarized to
        values {-1, 0, 1} based on proximity to those values.

        :param num_input_units: Number of input units of the network.
        :param num_hidden_layers: Number of hidden layers in the network.
        :param num_hidden_units: Number of units in each hidden layer.
        :param num_output_units: Number of output units of the network.
        :param bias_zero: Whether bias terms should be set to zero.
        :return: Neural network.
        """
        hyperparams = HyperparamsConfig(
            BATCH_SIZE=8,
            ALPHA=.15,
            EPSILON=1e-4,
            USE_BATCH_NORMALIZATION=False,
            HIDDEN_LAYERS_DIMS=[num_hidden_units] * num_hidden_layers,
            NUM_EPOCHS=1,
            DROPOUT_IN=0.,
            DROPOUT_HIDDEN=0.,
            BINARIZATION='BINARY',
            ZERO_THRESHOLD=0.05,
            STOCHASTIC=False,
            H=1.,
            W_LR_scale="Glorot",
            LR_start=0.,
            LR_fin=0.,
            LR_decay_type='exponential'
        )
        result = make_model(hyperparams, num_input_units, num_output_units)
        param_values = lasagne.layers.get_all_param_values(result['mlp'])
        for i in range(len(param_values)):
            if bias_zero and param_values[i].ndim == 1:
                # Bias terms correspond to 1D arrays;
                # while weights are 2D arrays.
                param_values[i] = np.zeros(param_values[i].shape)
            else:
                param_values[i] = (np.random.rand(*param_values[i].shape) * 2
                                   - 1.)
        lasagne.layers.set_all_param_values(result['mlp'], param_values)
        result['nn_output'] = lambda x: (theano.function(
                [result['input']], [result['test_output']])([x]))[0][0]
        result['hyperparams'] = hyperparams
        result['num_input_units'] = num_input_units
        result['num_output_units'] = num_output_units
        result['model_id'] = self.create_model_id()
        return result

    def create_NN(self,
                  num_input_units,
                  num_hidden_layers,
                  num_hidden_units,
                  num_output_units,
                  parameters):
        """
        Create a NN with provided parameter values.
        :param num_input_units: Number of input units of the network.
        :param num_hidden_layers: Number of hidden layers in the network.
        :param num_hidden_units: Number of units in each hidden layer.
        :param num_output_units: Number of output units of the network.
        :param parameters: Values of parameters of the neural network.
          In the format of list where parameter values should be ordered in
          the same way as theano get_all_param_values returns.
        :return: Neural network.
        """
        assert isinstance(parameters, list) or \
               (isinstance(parameters, np.ndarray) and parameters.ndim == 1)

        hyperparams = HyperparamsConfig(
            BATCH_SIZE=8,
            ALPHA=.15,
            EPSILON=1e-4,
            USE_BATCH_NORMALIZATION=False,
            HIDDEN_LAYERS_DIMS=[num_hidden_units] * num_hidden_layers,
            NUM_EPOCHS=1,
            DROPOUT_IN=0.,
            DROPOUT_HIDDEN=0.,
            BINARIZATION='TERNARY',
            ZERO_THRESHOLD=0.05,
            STOCHASTIC=False,
            H=1.,
            W_LR_scale="Glorot",
            LR_start=0.,
            LR_fin=0.,
            LR_decay_type='exponential'
        )
        result = make_model(hyperparams,
                            num_input_units,
                            num_output_units)
        random_param_values = lasagne.layers.get_all_param_values(result['mlp'])
        i = 0
        for layer_id in range(len(random_param_values)):
            layer_params = random_param_values[layer_id]
            if layer_params.ndim == 2:
                for row_id in range(layer_params.shape[0]):
                    for column_id in range(layer_params.shape[1]):
                        layer_params[row_id][column_id] = parameters[i]
                        i += 1
            else:
                for row_id in range(layer_params.shape[0]):
                    layer_params[row_id] = parameters[i]
                    i += 1
        if i != len(parameters):
            raise 'param_values not in the right format for this NN!'
        lasagne.layers.set_all_param_values(result['mlp'],
                                            random_param_values)
        result['nn_output'] = lambda x: (theano.function(
            [result['input']], [result['test_output']])([x]))[0][0]
        result['hyperparams'] = hyperparams
        result['num_input_units'] = num_input_units
        result['num_output_units'] = num_output_units
        result['model_id'] = self.create_model_id()
        return result

    def generate_input_values(self, num_examples, num_inputs):
        """
        Generates random inputs in range [-1,1].

        :param num_examples: Number of examples to generate.
        :param num_inputs: Number of inputs per example.
        :return: Numpy array of inputs of shape (num_examples, num_inputs)
        """
        return np.random.rand(num_examples, num_inputs) * 2 - 1.

    def create_experiments_mathematica_file(
        self, NN, inputs, path, tmax, convergence_types):

        with open(path, 'w') as f:
            experiment_name = 'test'
            translate_to_crn(NN['mlp'], experiment_name, f)
            translate_convergence_experiment(
                name=experiment_name,
                X=inputs,
                nn_output=NN['nn_output'],
                num_output_units=NN['num_output_units'],
                last_layer_num=len(NN['hyperparams'].HIDDEN_LAYERS_DIMS),
                f=f,
                tmax=tmax,
                convergence_types=convergence_types)
        # Add execute permissions to created mathematica file.
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def compute_convergence(self, NN, inputs, convergence_types):
        """
        :param NN: Neural Network.
        :param inputs: Inputs on which to run neural network.
        :convergence_types: Types of convergence times to compute.
        :return:
        """
        # TODO: There can be a check whether final simulation value reached
        # equals the output of NN.

        tmax = 500
        while True:
            mathematica_file = os.path.join(self.tmp_dir, 'nn.wls')
            self.create_experiments_mathematica_file(
                NN, inputs, mathematica_file, tmax, convergence_types)

            # To check why shell=False is desirable look here:
            # https://stackoverflow.com/questions/4256107/running-bash-commands-in-python
            # https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
            # If you modify command and it doesn't work a quick way to make it work
            # would probably be to set shell to True.
            process = subprocess.Popen(mathematica_file.split(),
                                       stdout=subprocess.PIPE,
                                       shell=False,
                                       cwd=self.tmp_dir)
            output, error = process.communicate()

            if error:
                print('WARNING: Error when executing mathematica file: {}'
                      .format(error))

            results_file = os.path.join(self.tmp_dir,
                                        'test_convergence_times.csv')
            results = []
            failure = False

            with open(results_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        results.append(
                            [float(x) for x in line.strip().split(',')])
                    except ValueError:
                        failure = True
                        break

            if failure:
                if tmax >= 4000.:
                    print('WARNING: cannot compute convergence time for large tmax.')
                tmax *= 2
                continue

            # Number of results should be equal to number of inputs.
            assert len(results) == inputs.shape[0]
            return np.array(results)

    def get_NN_params_flattened(self, NN):
        """Get NN parameters flattened in 1D."""
        param_values = lasagne.layers.get_all_param_values(NN['mlp'])
        result = []
        for layer_params in param_values:
            result.extend(layer_params.flatten().tolist())
        return np.array(result)

    def run_experiments(self,
                        num_input_units,
                        num_hidden_layers_configs,
                        num_hidden_units_configs,
                        num_output_units,
                        examples_per_model,
                        models_per_architecture,
                        callback):
        """
        Runs convergence time experiments.

        :num_input_units: Number of input units of the network.
        :num_hidden_layers_configs: List containing different configurations
          for number of hidden layers.
        :num_hidden_units_configs: List containing different configurations
          for number of units in hidden layers.
        :num_output_units: Number of output units of the network.
        :examples_per_model: How many input examples to generate per model.
        :models_per_architecture: How many models to create per each
          architecture choice.
        :callback: Callback providing ExperimentResult
        """
        assert isinstance(num_hidden_layers_configs, list) or \
               isinstance(num_hidden_layers_configs, np.ndarray)
        for i in range(len(num_hidden_layers_configs) - 1):
            assert num_hidden_layers_configs[i] < num_hidden_layers_configs[i + 1]

        assert isinstance(num_hidden_units_configs, list) or \
               isinstance(num_hidden_units_configs, np.ndarray)
        for i in range(len(num_hidden_units_configs) - 1):
            assert num_hidden_units_configs[i] < num_hidden_units_configs[i + 1]

        for config in itertools.product(num_hidden_layers_configs,
                                        num_hidden_units_configs,
                                        [0] * models_per_architecture):
            num_hidden_layers, num_hidden_units, _ = config
            NN = self.generate_random_NN(num_input_units,
                                         num_hidden_layers,
                                         num_hidden_units,
                                         num_output_units)
            inputs = self.generate_input_values(examples_per_model, num_input_units)
            convergence_time = self.compute_convergence(NN, inputs)
            result = ExperimentResult(
                model_id=NN['model_id'],
                num_input_units=num_input_units,
                num_output_units=num_output_units,
                num_hidden_layers=num_hidden_layers,
                num_hidden_units=num_hidden_units,
                convergence_time=convergence_time,
                inputs=inputs,
                parameters=None,
                date=datetime.now().strftime("%d/%m/%Y_%H:%M:%S"))
            callback(result)

    def run_experiments_random(self,
                        num_input_units,
                        num_hidden_layers_configs,
                        num_hidden_units_configs,
                        num_output_units,
                        examples_per_model,
                        total_models,
                        callback,
                        convergence_types):
        """
        Runs convergence time experiments, where for each experiment
        configuration is chosen at random. Total of :total_experiments
        experiments is run.

        :num_input_units: Number of input units of the network.
        :num_hidden_layers_configs: List containing different configurations
          for number of hidden layers.
        :num_hidden_units_configs: List containing different configurations
          for number of units in hidden layers.
        :num_output_units: Number of output units of the network.
        :examples_per_model: How many input examples to generate per model.
        :total_models: How many models to create in total.
        :callback: Callback providing ExperimentResult
        :convergence_types: Types of convergence times to compute.
        """
        assert isinstance(num_hidden_layers_configs, list) or \
               isinstance(num_hidden_layers_configs, np.ndarray)
        for i in range(len(num_hidden_layers_configs) - 1):
            assert num_hidden_layers_configs[i] < num_hidden_layers_configs[i + 1]

        assert isinstance(num_hidden_units_configs, list) or \
               isinstance(num_hidden_units_configs, np.ndarray)
        for i in range(len(num_hidden_units_configs) - 1):
            assert num_hidden_units_configs[i] < num_hidden_units_configs[i + 1]

        for _ in range(total_models):
            num_hidden_layers = random.choice(num_hidden_layers_configs)
            num_hidden_units = random.choice(num_hidden_units_configs)
            NN = self.generate_random_NN(num_input_units,
                                         num_hidden_layers,
                                         num_hidden_units,
                                         num_output_units)
            inputs = self.generate_input_values(examples_per_model,
                                                num_input_units)
            convergence_time = self.compute_convergence(NN,
                                                        inputs,
                                                        convergence_types)
            result = ExperimentResult(
                model_id=NN['model_id'],
                num_input_units=num_input_units,
                num_output_units=num_output_units,
                num_hidden_layers=num_hidden_layers,
                num_hidden_units=num_hidden_units,
                convergence_time=convergence_time,
                inputs=inputs,
                parameters=None,
                date=datetime.now().strftime("%d/%m/%Y_%H:%M:%S"))
            callback(result)


def write_result(csv_writer,
                 experiment_result,
                 examples_per_model):
    for i in range(examples_per_model):
        assert experiment_result.inputs.shape == (examples_per_model,
                                                  experiment_result.num_input_units)
        assert experiment_result.convergence_time.shape[0] == examples_per_model

        # IDEA: There can be 2 tables.
        # One that contains parameters for NN for each model_id.
        # Other that contains multiple experiments per each model.

        row = [experiment_result.model_id,
               experiment_result.num_input_units,
               experiment_result.num_output_units,
               experiment_result.num_hidden_layers,
               experiment_result.num_hidden_units]
        row.extend(experiment_result.convergence_time[i].tolist())
        row.extend(['-1', '-1', experiment_result.date])

        csv_writer.writerow(row)


def experiment(out_file,
               convergence_types):
    with open(out_file, 'w') as f:
        w = csv.writer(f)
        header = ['model_id',
                  'num_input_units',
                  'num_output_units',
                  'num_hidden_layers',
                  'num_hidden_units']
        header.extend(convergence_types)
        header.extend([
            'inputs',
            'parameters',
            'date'
        ])
        w.writerow(header)

        examples_per_model = 20
        ConvergenceExperiment().run_experiments_random(
            num_input_units=2,
            num_hidden_layers_configs=[1, 2, 3, 4, 5],
            num_hidden_units_configs=[2, 4, 8, 16],
            num_output_units=2,
            examples_per_model=examples_per_model,
            total_models=696969,
            callback=lambda result: write_result(
                csv_writer=w,
                experiment_result=result,
                examples_per_model=examples_per_model),
            convergence_types=convergence_types)


def plot_results(csv_file, convergence_column_name, out_dir):
    num_input_units = 2
    num_output_units = 2

    for num_hidden_units in [2, 4, 8, 16]:

        df = pd.read_csv(csv_file)

        df = df.loc[df['num_input_units'] == num_input_units]
        df = df.loc[df['num_output_units'] == num_output_units]
        df = df.loc[df['num_hidden_units'] == num_hidden_units]

        if len(df) == 0:
            continue

        # https://matplotlib.org/3.1.1/gallery/recipes/fill_between_alpha.html
        num_hidden_layers_configs = np.sort(df['num_hidden_layers'].unique())
        data = []
        for num_hidden_layers in num_hidden_layers_configs:
            df_fixed_layers = df.loc[df['num_hidden_layers'] == num_hidden_layers]
            convergence_times = df_fixed_layers[convergence_column_name].to_numpy()
            data.append(convergence_times)

        mu = []
        sigma = []
        for i in range(len(data)):
            mu.append(data[i].mean())
            sigma.append(data[i].std())
        mu = np.array(mu)
        sigma = np.array(sigma)

        fig, ax = plt.subplots()
        ax.plot(num_hidden_layers_configs, mu, lw=2,
                label='{} hidden_units'.format(num_hidden_units), color='blue')
        ax.fill_between(num_hidden_layers_configs,
                        mu + sigma, mu - sigma, facecolor='blue', alpha=0.25)
        ax.legend(loc='upper left')
        ax.set_xlabel('number of hidden layers')
        ax.set_ylabel('convergence time')
        plt.xticks(num_hidden_layers_configs)
        ax.grid()
        # plt.show()
        pathlib2.Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/hidden{}'.format(out_dir, num_hidden_units))
        plt.close()


def box_plot(csv_file, convergence_column_name):
    for model_id in range(100):
        df = pd.read_csv(csv_file)

        df = df.loc[df['model_id'] == model_id]

        convergence_times = list(df[convergence_column_name])

        data = np.array(convergence_times)

        fig, ax = plt.subplots()
        bp = ax.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
        ax.grid()
        # plt.show()
        plt.savefig(
            '/home/vasic/Downloads/sastanak/box_model{}'.format(model_id))
        plt.close()


def reconstruct_experiment(csv_file, result_id):
    df = pd.read_csv(csv_file)
    relevant_row = df.iloc[result_id]
    experiment = ConvergenceExperiment()
    parameters_string = relevant_row['parameters']
    parameters = np.fromstring(
        parameters_string[1:len(parameters_string) - 1],
        sep=' ',
        dtype=float)
    NN = experiment.create_NN(
        num_input_units=relevant_row['num_input_units'],
        num_output_units=relevant_row['num_output_units'],
        num_hidden_units=relevant_row['num_hidden_units'],
        num_hidden_layers=relevant_row['num_hidden_layers'],
        parameters=parameters)
    inputs_string = relevant_row['inputs']
    inputs = np.fromstring(
        inputs_string[1:len(inputs_string) - 1],
        sep=' ',
        dtype=float
    )
    convergence_time = experiment.compute_convergence(
        NN,
        np.array([inputs])
    )
    assert relevant_row['convergence_time'] == convergence_time


def main():
    # convergence_types = [
    #     'E-0.07',
    #     'F-0.9',
    #     'CE-0.01',
    #     'CF-0.95',
    #     'O',
    #     'OutputsSummed'
    # ]
    # convergence_types = ['OutputsSummed']
    convergence_types = ['F-0.9']
    out_dir = 'data/experiments/convergence/results/randomNNs'

    time_now = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    out_file = os.path.join(out_dir, '{}.csv'.format(time_now))
    experiment(out_file, convergence_types)

    # name = '2021.11.21_12:12:42'
    # name = '2021.11.26_18:33:04'
    # name = 'outputsSummedMerged'
    # csv_file = os.path.join(out_dir, '{}.csv'.format(name))
    # for convergence_type in convergence_types:
    #     out_dir = '/home/vasic/Downloads/sastanak/{}'.format(convergence_type)
    #     plot_results(csv_file, convergence_type, out_dir)
    # box_plot(out_file)

    # csv_file = os.path.join(out_dir, '2021.11.15_13:52:44.csv')
    # reconstruct_experiment(csv_file, result_id=0)


if __name__ == '__main__':
    main()
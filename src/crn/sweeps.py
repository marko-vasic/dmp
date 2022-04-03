import argparse
import math

import numpy as np

from crn import disease
from crn import pattern_formation
from crn import mnist
from crn.hyperparams import HyperparamsConfig
from crn.pattern_configs import Pattern
from crn.pattern_configs import get_pattern_configs

random_generator = np.random.RandomState(seed=None)

hyperparamsDefaultConfig = HyperparamsConfig(
    NUM_EPOCHS=250,
    USE_BATCH_NORMALIZATION=True,
    BATCH_SIZE=16,
    LR_start=1e-4,
    LR_fin=3e-7,
    LR_decay_type='exponential',
    DROPOUT_IN=0.,
    DROPOUT_HIDDEN=0.,
    HIDDEN_LAYERS_DIMS=[16],
    ALPHA=.15,
    EPSILON=1e-4,
    BINARIZATION='BINARY',
    ZERO_THRESHOLD=0.15,
    STOCHASTIC=False,
    H=1.,
    W_LR_scale="Glorot"
)


class DiscreteHyperparam(object):
    def __init__(self, values):
        self.values = values


class ContinuousHyperparam(object):
    def __init__(self, range_low, range_high, drawing='uniform'):
        """
        Parameters:
          drawing: Approach using for drawing a number in the range,
            supported values are 'unifrom' and 'geometric'.
        """
        self.range_low = range_low
        self.range_high = range_high
        self.drawing = drawing


class RandomSearch(object):
    def __init__(self, hyperparamsSpecs):
        self.hyperparamsSpecs = hyperparamsSpecs

    def pick(self):
        results = {}
        for name, hyperparam in self.hyperparamsSpecs.items():
            if isinstance(hyperparam, DiscreteHyperparam):
                results[name] = random_generator.choice(hyperparam.values)
            elif isinstance(hyperparam, ContinuousHyperparam):
                if hyperparam.drawing == 'uniform':
                    results[name] = random_generator.uniform(
                        low=hyperparam.range_low,
                        high=hyperparam.range_high,
                        size=1)[0]
                elif hyperparam.drawing == 'exponential':
                    low = math.log(hyperparam.range_low, 10)
                    high = math.log(hyperparam.range_high, 10)
                    number = random_generator.uniform(
                        low=low,
                        high=high,
                        size=1)[0]
                    results[name] = math.pow(10, number)
                else:
                    raise Exception("Unk drawing.")
            else:
                raise Exception("Unk hyperparam type.")
        return results


def config_valid(config):
    return config.LR_start >= config.LR_fin


def create_hyperparameter_specs(subject):
    hyperparamsSpecs = {}

    if subject == 'pattern_formation':
        hyperparamsSpecs['BINARIZATION'] = DiscreteHyperparam(
            ['BINARY', 'TERNARY'])
        hyperparamsSpecs['ZERO_THRESHOLD'] = DiscreteHyperparam(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
        hyperparamsSpecs['USE_BATCH_NORMALIZATION'] = DiscreteHyperparam(
            [False])
        hyperparamsSpecs['NUM_EPOCHS'] = DiscreteHyperparam(
            [5000, 10000, 20000, 30000, 50000])
        hyperparamsSpecs['BATCH_SIZE'] = DiscreteHyperparam([5, 51, 255])
        # parameters with name within <> are special ones, not directly present
        # in hyperparameters spec.
        hyperparamsSpecs['<HIDDEN_LAYERS_NUM>'] = DiscreteHyperparam(
            [4, 5])
        hyperparamsSpecs['<HIDDEN_LAYERS_SIZE>'] = DiscreteHyperparam(
            [8])
        hyperparamsSpecs['DROPOUT_IN'] = DiscreteHyperparam(
            [0., 0.05, 0.1])
        hyperparamsSpecs['DROPOUT_HIDDEN'] = DiscreteHyperparam(
            [0., 0.05, 0.1, 0.2])
        hyperparamsSpecs['LR_start'] = DiscreteHyperparam(
            [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
        hyperparamsSpecs['LR_fin'] = DiscreteHyperparam(
            [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 
             3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8])
        # lr_start_range = ContinuousHyperparam(range_low=1e-4, range_high=1e-1,
        #                                       drawing='exponential')
        # hyperparamsSpecs['LR_start'] = lr_start_range
        # lr_fin_range = ContinuousHyperparam(range_low=1e-8, range_high=1e-4,
        #                                     drawing='exponential')
        # hyperparamsSpecs['LR_fin'] = lr_fin_range
    elif subject == 'virus':
        hyperparamsSpecs['BINARIZATION'] = DiscreteHyperparam(
            ['BINARY', 'TERNARY'])
        hyperparamsSpecs['USE_BATCH_NORMALIZATION'] = DiscreteHyperparam(
            [False])
        hyperparamsSpecs['NUM_EPOCHS'] = DiscreteHyperparam([250, 500])
        hyperparamsSpecs['BATCH_SIZE'] = DiscreteHyperparam([16])
        hyperparamsSpecs['<HIDDEN_LAYERS_NUM>'] = DiscreteHyperparam([1, 2])
        hyperparamsSpecs['<HIDDEN_LAYERS_SIZE>'] = DiscreteHyperparam(
            [8, 16])
        hyperparamsSpecs['LR_start'] = DiscreteHyperparam(
            [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
        hyperparamsSpecs['LR_fin'] = DiscreteHyperparam(
            [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8])
    elif subject == 'mnist':
        hyperparamsSpecs['STOCHASTIC'] = DiscreteHyperparam(
            [True, False])
        hyperparamsSpecs['BINARIZATION'] = DiscreteHyperparam(
            ['BINARY', 'TERNARY'])
        hyperparamsSpecs['ZERO_THRESHOLD'] = DiscreteHyperparam(
            [0.10, 0.15, 0.20, 0.25, 0.35])
        hyperparamsSpecs['USE_BATCH_NORMALIZATION'] = DiscreteHyperparam(
            [False])
        hyperparamsSpecs['NUM_EPOCHS'] = DiscreteHyperparam(
            [250, 500, 1000, 3000])
        hyperparamsSpecs['BATCH_SIZE'] = DiscreteHyperparam([100, 256])
        # parameters with name within <> are special ones, not directly present
        # in hyperparameters spec.
        hyperparamsSpecs['<HIDDEN_LAYERS_NUM>'] = DiscreteHyperparam([1, 2, 3])
        hyperparamsSpecs['<HIDDEN_LAYERS_SIZE>'] = DiscreteHyperparam(
            [32, 64])
        hyperparamsSpecs['LR_start'] = DiscreteHyperparam(
            [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
        hyperparamsSpecs['LR_fin'] = DiscreteHyperparam(
            [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8])
    else:
        raise Exception('Unrecognized subject: ' + str(args.subject))
    return hyperparamsSpecs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str, required=True,
                        help='File where experiment results will be written to')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject for which experiments will be run.')
    parser.add_argument('--image_pattern', type=str,
                        help='Image pattern (used only in pattern formation datasets)')
    parser.add_argument('--num_sweeps', type=int, default=10000,
                        help='How many sweeps/experiments to perform')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rs = RandomSearch(create_hyperparameter_specs(args.subject))

    outF = open(args.out_file, "w")
    outF.write(','.join(hyperparamsDefaultConfig._asdict().keys()))
    outF.write(',num_nonzero_weights,num_weights,test_error,best_epoch\n')

    cnt = 0

    while True:
        picked = rs.pick()
        df = hyperparamsDefaultConfig._asdict()
        for key, val in picked.items():
            # Rewrite values with generated config.
            if key in df:
                df[key] = val
        if df['STOCHASTIC'] and df['BINARIZATION'] == 'TERNARY':
            # TODO: Rewrite nicer
            # Ternary weights with stochastic binarization not yet supported.
            continue
        hidden_num = picked['<HIDDEN_LAYERS_NUM>']
        hidden_size = picked['<HIDDEN_LAYERS_SIZE>']
        hidden_layers_dims = []
        for i in range(hidden_num):
            hidden_layers_dims.append(hidden_size)
        df['HIDDEN_LAYERS_DIMS'] = hidden_layers_dims

        config = HyperparamsConfig(**df)

        if not config_valid(config):
            continue

        if args.subject == 'pattern_formation':
            pattern_config, _ = get_pattern_configs(Pattern[args.image_pattern])
            model = pattern_formation.Model(config, pattern_config)
        elif args.subject == 'virus':
            model = disease.Model(config)
        elif args.subject == 'mnist':
            model = mnist.Model(config)
        else:
            raise Exception('Unrecognized subject: ' + str(args.subject))

        results = model.train(save_files=False)
        for value in config._asdict().values():
            if not isinstance(value, list):
                outF.write(str(value) + ',')
            else:
                outF.write('-'.join([str(x) for x in value]) + ',')
        outF.write(str(results['#total'] - results['#zero']) + ',')
        outF.write(str(results['#total']) + ',')
        outF.write(str(results['test_error']) + ',')
        outF.write(str(results['best_epoch']) + '\n')
        outF.flush()

        cnt += 1
        if cnt == args.num_sweeps:
            break
    outF.close()

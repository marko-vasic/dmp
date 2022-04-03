import collections

HyperparamsConfig = collections.namedtuple('HyperparamsConfig', [
    'BATCH_SIZE',
    # alpha is the exponential moving average factor
    'ALPHA',
    'EPSILON',
    'USE_BATCH_NORMALIZATION',
    'HIDDEN_LAYERS_DIMS',
    'NUM_EPOCHS',
    # Dropout parameters (0. means no dropout)
    'DROPOUT_IN',
    'DROPOUT_HIDDEN',
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


def compute_LR_decay(LR_start, LR_fin, NUM_EPOCHS, LR_decay_type):
    if LR_decay_type == 'exponential':
        return (LR_fin / LR_start) ** (1. / NUM_EPOCHS)
    else:
        raise Exception('Unsupported decay type: ' + str(LR_decay_type))
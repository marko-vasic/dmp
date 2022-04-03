from crn.hyperparams import HyperparamsConfig
from enum import Enum
import collections


class Pattern(Enum):
    OCILO = 1
    LONGHORN = 2
    HEART = 3
    TSHIRT = 4


class CoordinateSystem(Enum):
    # Cartesian coordinates with origin at a top left side of an image.
    CARTESIAN = 1
    # Polar coordinate system with origin at a center of an image.
    POLAR = 2
    # (x, y) coordinates where:
    # x coordinate is a distance from a center of an image
    #   (typically used for images that have symmetric left and right sides).
    # y coordinate is a distance from a top left side of an image.
    X_CENTERED = 3
    # x and y coordinates represent distance from a center of image.
    # used when image is symmetric left and right, as well as top and bottom.
    DIST_FROM_CENTER = 4
    # Includes (x1, y1, x2, y2) coordinates where
    # (x1, y1) are coordinates with origin at a top left of an image.
    # (x2, y2) are coordinates with origin at a bottom right of an image.
    CARTESIAN_DOUBLE = 5


class ColorSystem(Enum):
    BLACK_AND_WHITE = 1
    # TODO: Grayscale support still not fully implemented
    GRAYSCALE = 2


PatternConfig = collections.namedtuple('PatternConfig', [
    'COORDINATE_SYSTEM',
    'COLOR_SYSTEM',
    'MODEL_FILE',
    'IMAGE_FILE',
    'RECONSTRUCTED_IMAGE_FILE',
    'MATHEMATICA_FILE',
    'CRN_NAME'
])


def get_pattern_configs(pattern):
    if pattern == Pattern.HEART:
        pattern_config = PatternConfig(
            COORDINATE_SYSTEM=CoordinateSystem.X_CENTERED,
            COLOR_SYSTEM=ColorSystem.BLACK_AND_WHITE,
            MODEL_FILE='../data-repo/models/heart.pkl',
            MATHEMATICA_FILE='../data-repo/mathematica/heart.wls',
            IMAGE_FILE='../data-repo/datasets/pattern_formation/graphics/selected/heart_blackAndWhite.png',
            RECONSTRUCTED_IMAGE_FILE='../data-repo/reconstructed/heart-{}.png',
            CRN_NAME='HEART')
        hyperparams = HyperparamsConfig(
            BATCH_SIZE=255,
            ALPHA=.15,
            EPSILON=1e-4,
            USE_BATCH_NORMALIZATION=False,
            HIDDEN_LAYERS_DIMS=[8],
            NUM_EPOCHS=50000,
            DROPOUT_IN=0.,
            DROPOUT_HIDDEN=0.05,
            BINARIZATION='TERNARY',
            ZERO_THRESHOLD=0.05,
            STOCHASTIC=False,
            H=1.,
            W_LR_scale="Glorot",
            LR_start=0.1,
            LR_fin=3e-8,
            LR_decay_type='exponential'
        )
    else:
        raise Exception('Unrecognized pattern: {}'.format(pattern))

    return pattern_config, hyperparams

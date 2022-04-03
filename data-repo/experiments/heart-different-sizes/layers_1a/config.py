hyperparams_layers8 = HyperparamsConfig(
    BATCH_SIZE=255,
    ALPHA=.15,
    EPSILON=1e-4,
    USE_BATCH_NORMALIZATION=False,
    HIDDEN_LAYERS_DIMS=[8],
    NUM_EPOCHS=10000,
    DROPOUT_IN=0.,
    DROPOUT_HIDDEN=0.,
    BINARIZATION='TERNARY',
    ZERO_THRESHOLD=0.35,
    STOCHASTIC=False,
    H=1.,
    W_LR_scale="Glorot",
    LR_start=0.001,
    LR_fin=1E-05,
    LR_decay_type='exponential'
)

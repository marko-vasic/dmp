hyperparams_layers888 = HyperparamsConfig(
    BATCH_SIZE=5,
    ALPHA=.15,
    EPSILON=1e-4,
    USE_BATCH_NORMALIZATION=False,
    HIDDEN_LAYERS_DIMS=[8, 8, 8],
    NUM_EPOCHS=50000,
    DROPOUT_IN=0.,
    DROPOUT_HIDDEN=0.,
    BINARIZATION='TERNARY',
    ZERO_THRESHOLD=0.1,
    STOCHASTIC=False,
    H=1.,
    W_LR_scale="Glorot",
    LR_start=0.0003,
    LR_fin=1E-07,
    LR_decay_type='exponential'
)

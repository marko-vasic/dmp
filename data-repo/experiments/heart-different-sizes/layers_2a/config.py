hyperparams_layers88 = HyperparamsConfig(
    BATCH_SIZE=51,
    ALPHA=.15,
    EPSILON=1e-4,
    USE_BATCH_NORMALIZATION=False,
    HIDDEN_LAYERS_DIMS=[8, 8],
    NUM_EPOCHS=10000,
    DROPOUT_IN=0.,
    DROPOUT_HIDDEN=0.,
    BINARIZATION='TERNARY',
    ZERO_THRESHOLD=0.2,
    STOCHASTIC=False,
    H=1.,
    W_LR_scale="Glorot",
    LR_start=0.03,
    LR_fin=0.003,
    LR_decay_type='exponential'
)

hyperparams = HyperparamsConfig(
            BATCH_SIZE=51,
            ALPHA=.15,
            EPSILON=1e-4,
            USE_BATCH_NORMALIZATION=False,
            HIDDEN_LAYERS_DIMS=[8,8],
            NUM_EPOCHS=20000,
            DROPOUT_IN=0.05,
            DROPOUT_HIDDEN=0.,
            BINARIZATION='TERNARY',
            ZERO_THRESHOLD=0.25,
            STOCHASTIC=False,
            H=1.,
            W_LR_scale="Glorot",
            LR_start=0.1,
            LR_fin=1e-8,
            LR_decay_type='exponential'
        )

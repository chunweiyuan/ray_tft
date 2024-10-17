"""Constains relevants constants for TFT training/validation/prediction.
"""
from collections import OrderedDict

RANDOM_SEEDS = [7, 23, 28, 39, 43, 48,
                4, 6, 17, 25, 34, 47]  # lottery winning numbers

BATCH_SIZE = 70  # between 32 to 128.  Integer factor of # of data rows.
QUANTILES = OrderedDict(lower=0.025, median=0.5, upper=0.975)

# TFT constants that are empirically found to be optimal
LEARNING_RATE = 0.01
MIN_DELTA = 1e-3
Q_MIN_DELTA = 1e-3  # quantile loss func could use less stringent criterion
HIDDEN_SIZE = 64

LSTM_LAYERS = 1  # important?
ATTENTION_HEAD_SIZE = 4  # 4 is a good default, according to
# https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/__init__.py#L95

DROPOUT = 0.14015  # 0.1 - 0.3 are good values.

# Pytorch Lightning trainer constants
GRADIENT_CLIP_VAL = 0.0162781  # important?

# different choices of loss metric can be found in
# https://github.com/jdb78/pytorch-forecasting/tree/master/pytorch_forecasting/metrics

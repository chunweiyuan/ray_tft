"""Constains relevants constants for TFT training/validation/prediction.
"""
from collections import OrderedDict

BATCH_SIZE = 70  # between 32 to 128.  Integer factor of # of data rows.
N_VALIDATION_YEARS = 10  # number of validation years
QUANTILES = OrderedDict(lower=0.025, median=0.5, upper=0.975)
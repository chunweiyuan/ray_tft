This is an attempt to forecast age-specific fertility rate (ASFR)
using Temporal Fusion Transformer (from Pytorch Forecasting),
with Ray serving as the orchestrator to parallelize tasks across
CPUs.

Merely a illustration of feasibility.  Not intended for public use,
since the pipeline is highly bespoke.

The input data module is redacted away for privacy/security.

The tft_ray_test.py file reads in open-source data and should
work with simple `python tft_ray_test.py`, given the right
python environment (conda_list.txt) and cpu architecture:
https://github.com/ray-project/ray/issues/42135

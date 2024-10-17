"""Script for running TFT Fertility pipeline."""
import gc
import warnings

import lightning.pytorch as pl
import numpy as np
import ray
import xarray as xr
from fbd_research.fertility.asfr.tft.constants import QUANTILES
from fbd_research.fertility.asfr.tft.inputs import get_dataset
from fbd_research.fertility.asfr.tft.utils import (
    compute_validation_error, make_forecast, make_tft_and_trainer,
    make_time_series_dataset, make_training_validation_sets)
from pytorch_forecasting import Baseline
from pytorch_forecasting.metrics import RMSE, QuantileLoss
from scipy.special import expit, logit

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


# resource constants are semi-static: request at least this much from cluster.
NUM_CPUS = 10
MEMORY = 120  # total mem in GB.  At least NUM_CPUS * mem-per-task.

# highest level model parameters
n_trials = 10
point_metric = RMSE
n_train_years = 42  # the number of years available for training
n_val_years = 10

# relevant years
forecast_start = 2022
forecast_end = 2100

# first past year is determined via already-defined model parameters
past_start = forecast_start - n_train_years - n_val_years


# now get all the covariate and target data into one dataframe
df = get_dataset(past_start, forecast_start, forecast_end)

# a little bit of post-processing and data pruning
df["asfr"] = logit(df["asfr"])  # logit transform of single year asfr
df = df.fillna(0)  # TFT doesn't like NaNs

past_df = df.query(f"year_id >= {past_start} & year_id < {forecast_start}")

print(past_df.columns.tolist())
print(len(past_df))  # use a factor of this value to be your batch size

# training/validation only need past_df
train_dataset, val_dataset, train_dataloader, val_dataloader =\
    make_training_validation_sets(past_df,
                                  min_encoder_length=1,
                                  max_encoder_length=n_train_years,
                                  min_decoder_length=n_val_years,
                                  max_decoder_length=n_val_years,
                                  num_workers=NUM_CPUS)

del past_df
gc.collect()

baseline_error = compute_validation_error(val_dataloader,
                                          Baseline(),
                                          point_metric,
                                          expit)

print(f"baseline error is  {baseline_error}")

# make the prediction dataset.
# will attend to n_train_years past years to forecast all future years.
# need to redefine past_start because we no longer need validation years.
past_start = forecast_start - n_train_years
n_future_years = forecast_end - forecast_start + 1  # all future years

df = df.query(f"year_id >= {past_start}")

pred_dataset = make_time_series_dataset(
    df,
    min_encoder_length=n_train_years,
    max_encoder_length=n_train_years,
    min_prediction_length=n_future_years,
    max_prediction_length=n_future_years)

# now use Ray to distribute work amongst cores
runtime_env = {"env_vars": {"NCCL_SOCKET_IFNAME": "lo,docker0"}}

ray.init(include_dashboard=False,
         # log_to_driver=False,
         # runtime_env=runtime_env,
         num_cpus=NUM_CPUS,
         object_store_memory=0.9 * MEMORY * 1e9)

# some dimensions to parallelize over underneath
location_ids = df["location_id"].unique().tolist()
ages = df["age"].unique().tolist()

# initialize some results
das = []
val_das = []
val_errs = []

# make_forecast() and the dataloaders all have their own parallelizations.
for seed in range(n_trials):
    pl.seed_everything(seed)  # for reproducibility

    tft, trainer = make_tft_and_trainer(train_dataset,
                                        point_metric(),
                                        output_size=1)
    tft_q, trainer_q = make_tft_and_trainer(
        train_dataset,
        QuantileLoss(quantiles=list(QUANTILES.values())),
        output_size=len(QUANTILES))

    # fit network.  Dataloader uses multiprocessing with NUM_CPUS.
    trainer.fit(tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    # also fit the quantiles
    trainer_q.fit(tft_q,
                  train_dataloaders=train_dataloader,
                  val_dataloaders=val_dataloader)

    val_err = compute_validation_error(
        val_dataloader, tft, point_metric, expit)

    val_errs.append(val_err)

    # make_forecast() uses Ray for parallelization.
    val_da = expit(make_forecast(val_dataset, tft, tft_q,
                                 QUANTILES, location_ids, ages))
    val_da["seed"] = seed
    val_das.append(val_da)

    da = expit(make_forecast(pred_dataset, tft, tft_q, QUANTILES,
                             location_ids, ages))
    da["seed"] = seed
    das.append(da)

ray.shutdown()  # Ray is no longer needed beyond this point

val_error = np.mean(val_errs)
val_error_std = np.std(val_errs)

print(val_error, val_error_std)

val_da = xr.concat(val_das, dim="seed")
da = xr.concat(das, dim="seed")

del val_errs, val_das, das
gc.collect()

da.to_netcdf(f"asfr_{past_start}_{forecast_start}_{forecast_end}"
             f"_{point_metric.__name__}.nc")
val_da.to_netcdf(f"asfr_{past_start}_{forecast_start}_{forecast_end}"
                 f"_{point_metric.__name__}_val.nc")
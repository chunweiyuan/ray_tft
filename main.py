"""Script for running TFT Fertility pipeline."""
import warnings
from functools import reduce

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import ray
from fbd_research.fertility.asfr.tft.constants import (N_VALIDATION_YEARS,
                                                       QUANTILES)
from fbd_research.fertility.asfr.tft.inputs import get_dataset
from fbd_research.fertility.asfr.tft.utils import (
    compute_validation_error, make_forecast, make_tft_and_trainer,
    make_time_series_dataset, make_training_validation_sets)
from pytorch_forecasting import Baseline
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from scipy.special import logit

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


# resource constants mean we need to request at least as much from cluster.
NUM_CPUS = 10
MEMORY = 120  # GB.  In truth, each task only requires less than 5 GB.

# highest level model parameters
n_trials = 10
point_metric = RMSE
n_past_years = 42  # the number of years available for training

# relevant years
forecast_start = 2022
forecast_end = 2100

# first past year is determined via already-defined model parameters
past_start = forecast_start - n_past_years - N_VALIDATION_YEARS


# now get all the covariate and target data into one dataframe
df = get_dataset(past_start, forecast_start, forecast_end)

# a little bit of post-processing and data pruning
df["asfr"] = logit(df["asfr"])  # logit transform of single year asfr
df = df.fillna(0)  # TFT doesn't like NaNs

# now use Ray to distribute work amongst cores
runtime_env = {"env_vars": {"NCCL_SOCKET_IFNAME": "lo,docker0"}}

ray.init(include_dashboard=False,
         # log_to_driver=False,
         # runtime_env=runtime_env,
         num_cpus=NUM_CPUS,
         object_store_memory=0.9 * MEMORY * 1e9)


@ray.remote(num_cpus=1)
class Forecastor:

    def __init__(self,
                 df: pd.DataFrame,
                 n_past_years: int,
                 forecast_start: int,
                 n_validation_years: int):
        """
        """
        past_start = forecast_start - n_past_years - n_validation_years

        self.past_df = df.query(f"year_id >= {past_start} "
                                f"& year_id < {forecast_start}")

        train_dataset, val_dataset, train_dataloader, val_dataloader =\
            make_training_validation_sets(self.past_df, past_start,
                                          forecast_start, num_workers=0)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.location_ids = df["location_id"].unique().tolist()
        self.ages = df["age"].unique().tolist()

        n_future_years = forecast_end - forecast_start + 1  # all future years

        self.pred_df = df.query(f"year_id >= {forecast_start - n_past_years}")

        self.pred_dataset = make_time_series_dataset(
            self.pred_df,
            min_encoder_length=n_past_years,
            max_encoder_length=n_past_years,
            min_prediction_length=n_future_years,
            max_prediction_length=n_future_years)

    def train(self, seed: int):
        """
        """
        pl.seed_everything(seed)  # for reproducibility

        self.point_metric = RMSE
        self.quantile_metric = QuantileLoss(quantiles=list(QUANTILES.values()))

        self.tft, self.trainer =\
            make_tft_and_trainer(self.train_dataset,
                                 self.point_metric(),
                                 output_size=1)

        self.tft_q, self.trainer_q = make_tft_and_trainer(
            self.train_dataset,
            self.quantile_metric,
            output_size=len(QUANTILES))

        # fit network.  Dataloader uses multiprocessing with NUM_CPUS.
        self.trainer.fit(self.tft,
                         train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.val_dataloader)

        # also fit the quantiles
        self.trainer_q.fit(self.tft_q,
                           train_dataloaders=self.train_dataloader,
                           val_dataloaders=self.val_dataloader)

    def expit(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self) -> tuple:
        """
        """
        val_err = compute_validation_error(
            self.val_dataloader, self.tft, self.point_metric, self.expit)

        # make_forecast() uses Ray for parallelization.
        val_da = self.expit(make_forecast(
            self.val_dataset, self.tft, self.tft_q, QUANTILES,
            self.location_ids, self.ages))

        da = self.expit(make_forecast(self.pred_dataset, self.tft, self.tft_q,
                                      QUANTILES, self.location_ids, self.ages))

        return val_err, val_da, da


# initialize some results
das = []
val_das = []
val_errs = []

forecastors = []

for seed in range(n_trials):
    forecastor = Forecastor.remote(df, n_past_years, forecast_start,
                                   N_VALIDATION_YEARS)
    forecastor.train.remote(seed)
    forecastors.append(forecastor)

for forecastor in forecastors:
    val_err, val_da, da = forecastor.predict.remote()
    val_errs.append(val_err)
    val_das.append(val_das)
    das.append(das)

val_errs = ray.get(val_errs)
val_das = ray.get(val_das)
das = ray.get(das)

ray.shutdown()  # Ray is no longer needed beyond this point

val_error = np.mean(val_errs)
val_error_std = np.std(val_errs)

print(val_error, val_error_std)

val_da = reduce(lambda a, b: a + b, val_das) / n_trials
da = reduce(lambda a, b: a + b, das) / n_trials

da.to_netcdf(f"asfr_{past_start}_{forecast_start}_{forecast_end}"
             f"_{point_metric.__name__}.nc")
val_da.to_netcdf(f"asfr_{past_start}_{forecast_start}_{forecast_end}"
                 f"_{point_metric.__name__}_val.nc")
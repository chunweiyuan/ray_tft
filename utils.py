"""Contains TFT-specific routines for main pipeline.
"""
import warnings
from collections import OrderedDict
from typing import Callable, Union

import numpy as np
import pandas as pd
import ray
import xarray as xr
from fbd_research.fertility.asfr.tft.constants import BATCH_SIZE
from lightning.pytorch import Trainer
from pytorch_forecasting import (Baseline, TemporalFusionTransformer,
                                 TimeSeriesDataSet)
from pytorch_forecasting.metrics import Metric
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


def make_training_validation_sets(
        df: pd.DataFrame,
        min_encoder_length: int,
        max_encoder_length: int,
        min_decoder_length: int,
        max_decoder_length: int,
        num_workers: int) -> tuple[TimeSeriesDataSet, DataLoader]:
    """Set up training/validation datasets and their dataloaders.

    Args:
        df (pd.DataFrame): contains all training/validation data.
        min_encoder_length (int): min number of years for training.
        max_encoder_length (int): max number of years for training.
        min_decoder_length (int): min number of years for prediction.
        max_decoder_length (int): max number of years for prediction.
        num_workers (int): number of cores (aside from driver).
            Set to 0 if there's only one cpu available.
            An argument for pytorch lightning's dataloader, which
            uses python multiprocessing under the hood.

    Returns:
        (tuple[TimeSeriesDataSet, DataLoader]): training/validation
            datasets and dataloaders.
    """
    training_dataset = make_time_series_dataset(
        df,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_decoder_length,
        max_prediction_length=max_decoder_length)

    # create validation (set predict=True),
    # which means to predict the last max_prediction_length points in time
    # for each series
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        predict=True,
        stop_randomization=True,
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,  # both min/max are the same
        min_prediction_length=max_decoder_length,
        max_prediction_length=max_decoder_length)  # both min/max are the same

    # create dataloaders for model
    # num_workers=0 if only 1 cpu.
    # NOTE add batch_sample="synchronized" arg?
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=num_workers)

    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=num_workers)

    return (training_dataset, validation_dataset, train_dataloader,
            val_dataloader)


def make_tft_and_trainer(
        training_dataset: TimeSeriesDataSet,
        metric: Metric,
        output_size: int) -> tuple[TemporalFusionTransformer, Trainer]:
    """Make the TFT and Trainer objects.

    Contains all the hyperparameters learned from running this script,
    via optimize_hyperparameters().

    Args:
        training_dataset (TimeSeriesDataSet): the training datset.
        metric (Metric): loss function used for training.
            An example would be MAE() or RMSE().  Don't forget the
            parentheses.  For QuantileLoss, use something like
            QuantileLoss(quantiles=[0.025, 0.975])
        output_size (int): output_size param for tft.

    Returns:
        tuple([TemporalFusionTransformer, Trainer]):
            the TFT and Trainer objects.
    """
    # many of the hyperparameter values come from running train.py.

    trainer = Trainer(
        accelerator="auto",  # or "cpu"
        # clipping gradients is a hyperparameter important to prevent
        # divergence of the gradient for recurrent neural networks
        gradient_clip_val=0.0162781,  # number comes from running training.py
        # accumulate_grad_batches=4,
        # max_steps=5000,
        min_epochs=1,
        max_epochs=1,  # train for exactly 1 epoch
        enable_progress_bar=False,
        enable_checkpointing=False)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.010457789,  # most import hyperparameter
        hidden_size=17,  # 2nd most important hyperparameter
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.14015,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=15,  # set to <= hidden_size
        output_size=output_size,  # if QuantileLoss, this is # of quantiles
        loss=metric,  # maybe use RMSE() or MAE()
        optimizer="Adam",    # need this to avoid KeyError: 'radam_buffer'
        # reduce learning rate if no gain in validation loss after x epochs
        reduce_on_plateau_patience=4)

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    return tft, trainer


def make_time_series_dataset(
        df: pd.DataFrame,
        min_encoder_length: int,
        max_encoder_length: int,
        min_prediction_length: int,
        max_prediction_length: int) -> TimeSeriesDataSet:
    """Make and return the prediction dataset.

    This setup routine makes hard-coded assumptions about groups,
    static reals, time-varying columns, etc.


    Args:
        df (pd.DataFrame): the prediction dataframe, containing all past
            and future years that are used for attention & prediction.
        min_encoder_length (int): minimum encoder length.
        max_encoder_length (int): maximum encoder length.
        min_prediction_length (int): minimum prediction length.
        max_prediction_length (int): maximum prediction length.

    Returns:
        (TimeSeriesDataSet): the TFT dataset used for prediction
    """
    # these are the covariates known from past to future
    time_varying_known_reals = ["education", "met_need", "u5m", "urbanicity"]

    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="year_id",
        target="asfr",
        group_ids=["location_id", "region_id", "high_income", "age"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_reals=["location_id", "region_id", "high_income", "age"],
        static_categoricals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["asfr"],  # this is the forecast target
        target_normalizer=None,
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False)

    return dataset


def make_location_age_forecast(
        tft: TemporalFusionTransformer,
        dataset: TimeSeriesDataSet,
        location_id: int,
        age: int,
        quantiles: Union[OrderedDict, None] = None) -> xr.DataArray:
    """Forecast all future years given location and age.

    Args:
        tft (TemporalFusionTransformer): the trained tft
            ready to make forecasts.
        dataset (TimeSeriesDataSet): dataset used to attend
            to to make forecast.
        location_id (int): location id.
        age (int): age
        quantiles (Union[OrderedDict, None]): if using
            QuantileLoss loss function, then the last
            dim of the forecast output would be the
            quantiles.  The keys of the ordered dict
            would be the names of the quantiles.
            If None, then it's the mean.

    Returns:
        (xr.DataArray): dataarray with location_id, age,
            and year_id dimensions.
    """
    forecast = tft.predict(dataset.filter(
        lambda x: (x.location_id == location_id) & (x.age == age)),
        mode="raw",
        return_x=True)

    # need to inverse-transform the results from logit space
    future_years = np.asarray(forecast.x["decoder_time_idx"][0])
    # overwrite torch object because we no longer need it
    forecast = np.asarray(forecast.output.prediction[0])
    if quantiles is None:  # must be mean
        forecast = xr.DataArray(
            forecast, coords=[("year_id", future_years),
                              ("statistic", ["mean"])])
    else:  # must be quantiles
        forecast = xr.DataArray(
            forecast, coords=[("year_id", future_years),
                              ("statistic", list(quantiles.keys()))])
    forecast["location_id"] = location_id
    forecast["age"] = age

    return forecast


def make_forecast(dataset: TimeSeriesDataSet,
                  tft: TemporalFusionTransformer,
                  tft_q: TemporalFusionTransformer,
                  quantiles: OrderedDict,
                  location_ids: list,
                  ages: list) -> xr.DataArray:
    """Make forecast over provided locations and ages.

    Args:
        dataset (TimeSeriesDataSet): dataset used to attend
            to to make forecast.
        tft (TemporalFusionTransformer): the trained tft
            ready to make point forecasts.
        tft_q (TemporalFusionTransformer): the trained tft
            ready to make quantile forecasts.
        quantiles (OrderedDict): ordered dict of quantiles used
            by the QuantileLoss function.  The keys of the ordered dict
            would be the names of the quantiles.
        location_ids (list): location_ids to loop over.
        ages (list): ages to loop over.

    Returns:
        (xr.DataArray): dataarray with all location_ids, ages, and
            forecasted year_ids.
    """
    # need to loop over locations & ages:
    location_das = []
    for location_id in location_ids:

        # Ray remote call returns a "future" ref
        location_da = _location_forecast.remote(
            dataset, tft, tft_q, quantiles, location_id, ages)

        location_das.append(location_da)  # list of ray future refs

    location_das = ray.get(location_das)

    return xr.concat(location_das, dim="location_id")


@ray.remote(num_cpus=1)
def _location_forecast(dataset: TimeSeriesDataSet,
                       tft: TemporalFusionTransformer,
                       tft_q: TemporalFusionTransformer,
                       quantiles: OrderedDict,
                       location_id: int,
                       ages: list) -> xr.DataArray:
    """Given location_id, make point/quantile forecasts for all the ages.
    """
    age_das = []
    for age in ages:
        # first the mean forecast
        forecast = make_location_age_forecast(
            tft, dataset, location_id=location_id,
            age=age, quantiles=None)

        # now the quantiles forecast
        forecast_q = make_location_age_forecast(
            tft_q, dataset, location_id=location_id,
            age=age, quantiles=quantiles)  # back transform

        forecast = xr.concat([forecast, forecast_q],
                             dim="statistic")
        age_das.append(forecast)

    return xr.concat(age_das, dim="age")


@ray.remote(num_cpus=1)
def _concat(*arrays: tuple, **kwargs: dict) -> xr.DataArray:
    """Concatenate given arrays."""
    return xr.concat(arrays, **kwargs)


def compute_validation_error(val_dataloader: DataLoader,
                             model: Union[Baseline, TemporalFusionTransformer],
                             metric: Metric,
                             transform: Callable) -> float:
    """Compute validation error in transformed space.

    Args:
        val_dataloader (DataLoader): dataloader containing
            validation data.
        model (Union[Baseline, TemporalFusionTransformer]):
            either Baseline(), or trained TFT.
        metric (Metric): likely point metric such as RMSE or MAE.
        transform (Callable): a function compatible with torch's
            ndarray.  If no transform, just pass a lambda x: x.

    Returns:
        (float): a single floating point value for error.
    """
    predictions = model.predict(val_dataloader, return_y=True)
    error = metric()(transform(predictions.output),
                     transform(predictions.y[0].reshape(
                               tuple(predictions.output.shape))))
    return float(error)
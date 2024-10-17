"""Contains TFT-specific routines for main pipeline.
"""
import warnings
from collections import OrderedDict
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from fbd_research.fertility.asfr.tft.constants import BATCH_SIZE
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
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
        max_decoder_length: int) -> tuple[TimeSeriesDataSet, DataLoader]:
    """Set up training/validation datasets and their dataloaders.

    Note that this function can also be used to make training/test sets.

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
    train_dataset = make_time_series_dataset(
        df,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_decoder_length,
        max_prediction_length=max_decoder_length)

    # create validation (set predict=True),
    # which means to predict the last max_prediction_length points in time
    # for each series
    validation_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
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
    train_dataloader = train_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    return (train_dataset, validation_dataset, train_dataloader,
            val_dataloader)


def make_tft(
        train_dataset: TimeSeriesDataSet,
        metric: Metric,
        output_size: int,
        learning_rate: float,
        hidden_size: int,
        lstm_layers: int,
        attention_head_size: int,
        dropout: float) -> TemporalFusionTransformer:
    """Make temporal fusion transformer.

    https://github.com/jdb78/pytorch-forecasting/blob/master/
    pytorch_forecasting/models/temporal_fusion_transformer/__init__.py#L29

    Args:
        train_dataset (TimeSeriesDataSet): the training datset.
        metric (Metric): loss function used for training.
            An example would be MAE() or RMSE().  Don't forget the
            parentheses.  For QuantileLoss, use something like
            QuantileLoss(quantiles=[0.025, 0.975])
        output_size (int): output_size param for tft.
        learning_rate (float): learning rate of gradient descent step.
            Most important hyperparameter of TFT.
        hidden_size (int): size of internal vector representation of
            features.  2nd most important hyperparameter of TFT.
        lstm_layers (int): number of lstm layers.
            Empirically, this impacts performance as well.
        attention_head_size (int): number of attention heads.
            TFT source code says 4 is a good default.
        dropout (float): dropout rate.  Fraction of lstm connections
            omitted.  Might be better to be < 0.5.

    Returns:
        (TemporalFusionTransformer): temporal fusion transformer.
    """
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,  # most import hyperparameter
        hidden_size=hidden_size,  # 2nd most important hyperparameter
        lstm_layers=lstm_layers,  # seems to also impact performance
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=attention_head_size,  # TODO not yet grid-searched
        dropout=dropout,  # 0.1 - 0.3 are good values.  from tune.py
        hidden_continuous_size=hidden_size,  # usually set to <= hidden_size.
        output_size=output_size,  # if QuantileLoss, this is # of quantiles
        loss=metric,  # maybe use RMSE() or MAE()
        optimizer="Adam",    # need this to avoid KeyError: 'radam_buffer'
        # reduce learning rate if no gain in validation loss after x epochs
        reduce_on_plateau_patience=4)

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    return tft


def make_trainer(
        min_delta: float,
        gradient_clip_val: float,
        max_steps: int = -1,
        min_epochs: Optional[int] = None,
        max_epochs: Optional[int] = None,
        enable_progress_bar: Optional[bool] = False,
        enable_checkpointing: Optional[bool] = False,
        logger: Optional[bool] = False,
        log_every_n_steps: Optional[int] = 0) -> Trainer:
    """Make pytorch lighting trainer.

    https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/
    lightning/pytorch/trainer/trainer.py#L92

    Args:
        min_delta (float): minimum change of epoch validation error
            to keep on training.
        learning_rate (float): learning rate of gradient descent step.
            Most important hyperparameter of TFT.
        hidden_size (int): size of internal vector representation of
            features.  2nd most important hyperparameter of TFT.
        lstm_layers (int): number of lstm layers.
            Empirically, this impacts performance as well.
        max_steps (int): stop training after this number of steps.
            Disabled by default (-1). If ``max_steps = -1`` and
            ``max_epochs = None``, will default to ``max_epochs = 1000``.
            To enable infinite training, set ``max_epochs`` to ``-1``.
        min_epochs (Optional[int]): force training for at least these
            many epochs.  Disabled by default (None).
        max_epochs (Optional[int]): stop training once this number of epochs
            is reached.  Disabled by default (None).
            If both max_epochs and max_steps are not specified,
            defaults to ``max_epochs = 1000``.
            To enable infinite training, set ``max_epochs = -1``.
        enable_checkpointing (Optional[bool]): if ``True``, enable
            checkpointing. It will configure a default ModelCheckpoint callback
            if there is no user-defined ModelCheckpoint in
            :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`.
            Default: ``False``.  As long as training finishes, it doesn't
            seem to matter.
        enable_progress_bar (Optional[bool]): whether to enable to progress
            bar by default. Default: ``False``.
        logger (Optional[bool]): whether to enable Trainer logging.
            Could produce lots of log files.  Default: ``False``.
        log_very_n_steps (Optional[int])): defaults to 0 (no logging).

    Returns:
        (Trainer): trainer from pytorch lightning.
    """
    # Note that min/max_epochs supercede early-stopping,
    # so early-stop won't stop if < min_epochs, and stops when >= max_epochs.
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=min_delta,  # ~ min values
                                        patience=10,  # wait 10 epochs
                                        verbose=True,
                                        mode="min")

    # many of the hyperparameter values come from running tune.py
    trainer = Trainer(
        accelerator="cpu",
        # clipping gradients is a hyperparameter important to prevent
        # divergence of the gradient for recurrent neural networks
        gradient_clip_val=gradient_clip_val,
        max_steps=max_steps,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback],
        enable_progress_bar=enable_progress_bar,
        enable_checkpointing=enable_checkpointing,
        logger=logger,
        log_every_n_steps=log_every_n_steps)

    return trainer


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

    # these are the columns that identify a specific group
    group_ids = ["location_id", "age", "region_id", "high_income"]

    # don't want scaling of any variable.  For more info:
    # https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/data/timeseries.py#L311
    scalers = {x: None for x in group_ids + time_varying_known_reals}

    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="year_id",
        target="asfr",
        group_ids=group_ids,
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_reals=group_ids,
        static_categoricals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["asfr"],  # this is the forecast target
        target_normalizer=None,
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        scalers=scalers
    )

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
    forecast = np.asarray(forecast.output.prediction[0])  # shape = (n, 1)
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

        loc_da = xr.concat(age_das, dim="age")
        location_das.append(loc_da)

    return xr.concat(location_das, dim="location_id")


def compute_error(dataloader: DataLoader,
                  model: Union[Baseline, TemporalFusionTransformer],
                  metric: Metric,
                  transform: Callable) -> float:
    """Compute validation/test error in transformed space.

    Args:
        dataloader (DataLoader): validation/test dataloader containing
            training/test data.
        model (Union[Baseline, TemporalFusionTransformer]):
            either Baseline(), or trained TFT.
        metric (Metric): likely point metric such as RMSE or MAE.
        transform (Callable): a function compatible with torch's
            ndarray.  If no transform, just pass a lambda x: x.

    Returns:
        (float): a single floating point value for error.
    """
    predictions = model.predict(dataloader, return_y=True)
    error = metric()(transform(predictions.output),
                     transform(predictions.y[0].reshape(
                               tuple(predictions.output.shape))))
    return float(error)


def get_interpretation(
        dataset: TimeSeriesDataSet,
        tft: TemporalFusionTransformer,
        location_ids: list,
        ages: list) -> tuple[xr.DataArray]:
    """Get all interpretation scores.

    Args:
        tft (TemporalFusionTransformer): the trained/validated tft.
        dataset (TimeSeriesDataSet): typically the validation dataset.
        location_id (int): location id.
        age (int): age.

    Returns:
        tuple[xr.DataArray]: attention_score is a dataarray with
            location_id, age, and year_id dims.  The static/encoder/decoder
            scores have location_id, age, and varible dims.
    """
    # need to loop over locations & ages:
    loc_att_scores = []
    loc_static_scores = []
    loc_encoder_scores = []
    loc_decoder_scores = []

    for location_id in location_ids:

        age_att_scores = []
        age_static_scores = []
        age_encoder_scores = []
        age_decoder_scores = []

        for age in ages:
            # first the mean forecast
            attention_score, static_score, encoder_score, decoder_score =\
                get_location_age_interpretation(
                    tft, dataset, location_id=location_id, age=age)

            age_att_scores.append(attention_score)
            age_static_scores.append(static_score)
            age_encoder_scores.append(encoder_score)
            age_decoder_scores.append(decoder_score)

        loc_att_scores.append(xr.concat(age_att_scores, dim="age"))
        loc_static_scores.append(xr.concat(age_static_scores, dim="age"))
        loc_encoder_scores.append(xr.concat(age_encoder_scores,
                                            dim="age"))
        loc_decoder_scores.append(xr.concat(age_decoder_scores,
                                            dim="age"))

    attention_score = xr.concat(loc_att_scores, dim="location_id")
    static_score = xr.concat(loc_static_scores, dim="location_id")
    encoder_score = xr.concat(loc_encoder_scores, dim="location_id")
    decoder_score = xr.concat(loc_decoder_scores, dim="location_id")

    return attention_score, static_score, encoder_score, decoder_score


def get_location_age_interpretation(
        tft: TemporalFusionTransformer,
        dataset: TimeSeriesDataSet,
        location_id: int,
        age: int) -> tuple[xr.DataArray]:
    """Get location-age interpretation scores from TFT and validation dataset.

    Args:
        tft (TemporalFusionTransformer): the trained tft
            ready to make forecasts.
        dataset (TimeSeriesDataSet): typically the validation dataset.
        location_id (int): location id.
        age (int): age.

    Returns:
        tuple[xr.DataArray]: attention_score is a dataarray with
            location_id/age and year_id dim.  The static/encoder/decoder
            scores have location_id/age and varible dim.
    """
    prediction = tft.predict(dataset.filter(
        lambda x: (x.location_id == location_id) & (x.age == age)),
        mode="raw",
        return_x=True)

    # first get the encoder year_ids
    encoder_length = int(prediction.x["encoder_lengths"][0])
    encoder_time_start =\
        int(prediction.x['decoder_time_idx'][0].min()) - encoder_length
    encoder_times = range(encoder_time_start,
                          encoder_time_start + encoder_length)

    # get the interpretation
    interpretation = tft.interpret_output(prediction.output, reduction="sum")

    # the attention scores in time (they sum to 1)
    attention_score = xr.DataArray(
            np.asarray(interpretation["attention"]),
            coords=[("year_id", encoder_times)])
    attention_score["location_id"] = location_id
    attention_score["age"] = age

    # the static variables significance.  Also sum to 1.
    static_variables = tft.static_variables  # list of variable names
    static_score = xr.DataArray(
        np.asarray(interpretation["static_variables"]),
        coords=[("variable", static_variables)])
    static_score["location_id"] = location_id
    static_score["age"] = age

    # encoder scores.  they sum to 1.
    encoder_variables = tft.encoder_variables  # list of variable names
    encoder_score = xr.DataArray(
            np.asarray(interpretation["encoder_variables"]),
            coords=[("variable", encoder_variables)])
    encoder_score["location_id"] = location_id
    encoder_score["age"] = age

    # decoder scores.  they sum to 1.
    decoder_variables = tft.decoder_variables  # list of variable names
    decoder_score = xr.DataArray(
        np.asarray(interpretation["decoder_variables"]),
        coords=[("variable", decoder_variables)])
    decoder_score["location_id"] = location_id
    decoder_score["age"] = age

    return attention_score, static_score, encoder_score, decoder_score

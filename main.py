import gc
import warnings

import click
import lightning.pytorch as pl
import numpy as np
import xarray as xr
from fbd_research.fertility.asfr.tft.constants import (ATTENTION_HEAD_SIZE,
                                                       DROPOUT,
                                                       GRADIENT_CLIP_VAL,
                                                       HIDDEN_SIZE,
                                                       LEARNING_RATE,
                                                       LSTM_LAYERS, MIN_DELTA,
                                                       Q_MIN_DELTA, QUANTILES)
from fbd_research.fertility.asfr.tft.inputs import get_dataset
from fbd_research.fertility.asfr.tft.utils import (
    compute_error, get_interpretation, make_forecast, make_tft,
    make_time_series_dataset, make_trainer, make_training_validation_sets)
from pytorch_forecasting import Baseline
from pytorch_forecasting.metrics import RMSE, QuantileLoss
from scipy.special import expit, logit

warnings.filterwarnings("ignore")  # avoid printing out absolute paths


@click.command()
@click.option("--training-years", nargs=2, type=int, required=True,
              help="start/end training years: --training-years 1970 2009")
@click.option("--validation-years", nargs=2, type=int, required=True,
              help="start/end validation years: --validation-years 2010 2019")
@click.option("--test-years", nargs=2, type=int, required=True,
              help="start/end test years, such as --test-years 2020 2021")
@click.option("--forecast-years", nargs=2, type=int, required=True,
              help="start/end years of forecast.")
@click.option("--learning-rate", type=float, default=LEARNING_RATE,
              help="learning rate.")
@click.option("--min-delta", type=float, default=MIN_DELTA,
              help="min validation loss decrement for mean optimization.")
@click.option("--q-min-delta", type=float, default=Q_MIN_DELTA,
              help="min validation loss decrement for quantile optimization.")
@click.option("--hidden-size",  type=int, default=HIDDEN_SIZE,
              help="the length of the latent vector representation.")
@click.option("--attention-head-size", type=int, default=ATTENTION_HEAD_SIZE,
              help="number of attention heads.")
@click.option("--seed", type=int, default=0,
              help="random number seed.")
def main(training_years: tuple,
         validation_years: tuple,
         test_years: tuple,
         forecast_years: tuple,
         learning_rate: float,
         min_delta: float,
         q_min_delta: float,
         hidden_size: int,
         attention_head_size: int,
         seed: int):
    """The main orchestrator of the TFT fertility pipeline."""

    # highest level parameters
    point_metric = RMSE
    training_start, training_end = training_years
    validation_start, validation_end = validation_years
    test_start, test_end = test_years
    forecast_start, forecast_end = forecast_years

    # determine the sizes for training/validation/test data sets
    n_val_years = validation_end - validation_start + 1
    n_train_years = training_end - training_start + 1
    n_test_years = test_end - test_start + 1
    n_future_years = forecast_end - forecast_start + 1

    # now get all the covariate and target data into one dataframe
    df = get_dataset(training_start, forecast_end, national_only=True)

    # a little bit of post-processing and data pruning
    df["asfr"] = logit(df["asfr"])  # logit transform of single year asfr
    df = df.fillna(0)  # TFT doesn't like NaNs

    train_val_df = df.query(f"year_id >= {training_start} & "
                            f"year_id <= {validation_end}")

    print(train_val_df.columns.tolist())
    print(len(train_val_df))  # use a factor of this value to be batch size

    # training/validation only need past_df
    train_dataset, val_dataset, train_dataloader, val_dataloader =\
        make_training_validation_sets(train_val_df,
                                      min_encoder_length=n_train_years,
                                      max_encoder_length=n_train_years,
                                      min_decoder_length=n_val_years,
                                      max_decoder_length=n_val_years)

    del train_val_df
    gc.collect()

    baseline_error = compute_error(val_dataloader,
                                   Baseline(),
                                   point_metric,
                                   expit)

    print(f"baseline error is  {baseline_error}")

    # make a test dataset that has training & test years in succession.
    # will train on n_train_years past years to forecast n_test_years.
    # need to redefine training_start because the years are moving up.
    test_training_start = test_end - n_test_years - n_train_years + 1
    train_test_df = df.query(f"year_id >= {test_training_start} & "
                             f"year_id <= {test_end}")

    _, test_dataset, _, test_dataloader =\
        make_training_validation_sets(train_test_df,
                                      min_encoder_length=n_train_years,
                                      max_encoder_length=n_train_years,
                                      min_decoder_length=n_test_years,
                                      max_decoder_length=n_test_years)

    del train_test_df
    gc.collect()

    # make the prediction dataset.
    # will train on n_train_years past years to forecast all future years.
    # need to redefine train_start because we no longer need validation years.
    pred_training_start = forecast_start - n_train_years

    pred_df = df.query(f"year_id >= {pred_training_start} & "
                       f"year_id <= {forecast_end}")

    pred_dataset = make_time_series_dataset(
        pred_df,
        min_encoder_length=n_train_years,
        max_encoder_length=n_train_years,
        min_prediction_length=n_future_years,
        max_prediction_length=n_future_years)

    del pred_df
    gc.collect()

    # initalialize these iterables
    das = []
    test_das = []
    test_errs = []
    attention_scores = []
    static_scores = []
    encoder_scores = []
    decoder_scores = []

    location_ids = df["location_id"].unique().tolist()
    ages = df["age"].unique().tolist()

    pl.seed_everything(seed)  # for reproducibility

    tft = make_tft(train_dataset,
                   point_metric(),
                   output_size=1,
                   learning_rate=learning_rate,
                   hidden_size=hidden_size,
                   lstm_layers=LSTM_LAYERS,
                   attention_head_size=attention_head_size,
                   dropout=DROPOUT)

    trainer = make_trainer(
        min_delta=min_delta,
        gradient_clip_val=GRADIENT_CLIP_VAL)

    # make tft and trainer for the quantiles as well
    tft_q = make_tft(train_dataset,
                     QuantileLoss(quantiles=list(QUANTILES.values())),
                     output_size=len(QUANTILES),
                     learning_rate=learning_rate,
                     hidden_size=hidden_size,
                     lstm_layers=LSTM_LAYERS,
                     attention_head_size=attention_head_size,
                     dropout=DROPOUT)

    trainer_q = make_trainer(
        min_delta=q_min_delta,
        gradient_clip_val=GRADIENT_CLIP_VAL)

    # fit networks
    trainer.fit(tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    # also fit the quantiles
    trainer_q.fit(tft_q,
                  train_dataloaders=train_dataloader,
                  val_dataloaders=val_dataloader)

    # calcualte the RMSE error of our validation set prediction
    test_err = compute_error(
        test_dataloader, tft, point_metric, expit)

    test_err = xr.DataArray([test_err], coords=[("seed", [seed])])

    test_errs.append(test_err)

    test_loss = trainer.test(model=tft,
                             dataloaders=test_dataloader,
                             ckpt_path=None,
                             verbose=False)
    test_loss = test_loss[0]
    test_loss = xr.DataArray([list(test_loss.values())],
                             coords=[("seed", [seed]),
                                     ("type", list(test_loss.keys()))])

    test_da = expit(make_forecast(test_dataset, tft, tft_q,
                                  QUANTILES, location_ids, ages))
    test_da["seed"] = seed
    test_das.append(test_da)

    da = expit(make_forecast(pred_dataset, tft, tft_q, QUANTILES,
                             location_ids, ages))
    da["seed"] = seed
    das.append(da)

    attention_score, static_score, encoder_score, decoder_score =\
        get_interpretation(val_dataset, tft, location_ids, ages)

    attention_score["seed"] = seed
    static_score["seed"] = seed
    encoder_score["seed"] = seed
    decoder_score["seed"] = seed

    attention_scores.append(attention_score)
    static_scores.append(static_score)
    encoder_scores.append(encoder_score)
    decoder_scores.append(decoder_score)

    # average over trials
    test_error = np.mean(test_errs)
    test_error_std = np.std(test_errs)

    print(test_error, test_error_std)

    def assign_attributes(arr: xr.DataArray) -> xr.DataArray:
        """Assign local variable attributes to input array"""
        arr["learning_rate"] = learning_rate
        arr["hidden_size"] = hidden_size
        arr["lstm_layers"] = LSTM_LAYERS
        arr["attention_head_size"] = attention_head_size
        arr["min_delta"] = min_delta
        arr["q_min_delta"] = q_min_delta
        arr["training_years"] = f"{training_years[0]}-{training_years[1]}"
        arr["validation_years"] =\
            f"{validation_years[0]}-{validation_years[1]}"
        arr["test_years"] = f"{test_years[0]}-{test_years[1]}"
        arr["forecast_years"] = f"{forecast_years[0]}-{forecast_years[1]}"

        return arr

    da = assign_attributes(xr.concat(das, dim="seed"))
    test_da = assign_attributes(xr.concat(test_das, dim="seed"))
    test_err = assign_attributes(xr.concat(test_errs, dim="seed"))

    del test_errs, test_das, das
    gc.collect()

    da.to_netcdf(f"asfr_{pred_training_start}_{forecast_start}_{forecast_end}_"
                 f"lr{learning_rate}_h{hidden_size}_{min_delta}_{seed}.nc")

    test_da.to_netcdf(f"asfr_test_{test_training_start}_{test_start}_"
                      f"{test_end}_lr{learning_rate}_h{hidden_size}_"
                      f"{min_delta}_{seed}.nc")

    test_err.to_netcdf(f"asfr_err_{test_training_start}_{test_start}_"
                       f"{test_end}_lr{learning_rate}_h{hidden_size}_"
                       f"{min_delta}_{seed}.nc")

    test_loss.to_netcdf(f"asfr_test_loss_{test_training_start}_{test_start}_"
                        f"{test_end}_lr{learning_rate}_h{hidden_size}_"
                        f"{min_delta}_{seed}.nc")

    del da, test_da, test_err, test_loss
    gc.collect()

    attention_scores =\
        assign_attributes(xr.concat(attention_scores, dim="seed"))
    static_scores = assign_attributes(xr.concat(static_scores, dim="seed"))
    encoder_scores = assign_attributes(xr.concat(encoder_scores, dim="seed"))
    decoder_scores = assign_attributes(xr.concat(decoder_scores, dim="seed"))

    attention_scores.to_netcdf(
        f"attention_{training_start}_{validation_start}_{validation_end}"
        f"_{point_metric.__name__}.nc")

    static_scores.to_netcdf(
        f"static_{training_start}_{validation_start}_{validation_end}"
        f"_{point_metric.__name__}.nc")

    encoder_scores.to_netcdf(
        f"encoder_{training_start}_{validation_start}_{validation_end}"
        f"_{point_metric.__name__}.nc")

    decoder_scores.to_netcdf(
        f"decoder_{training_start}_{validation_start}_{validation_end}"
        f"_{point_metric.__name__}.nc")


if __name__ == "__main__":
    main()

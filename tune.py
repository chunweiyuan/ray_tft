"""A module to obtain high-level TFT hyperparameters.

Running the script with ``python tune.py`` performs calculations
to find suitable hyperparameters, given the input parameters.

Takes a long time (~30h).  Can be lessened with fewer max_epochs.
See https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/tuning.py
"""
import gc
import pickle
import warnings

from fbd_research.fertility.asfr.tft.inputs import get_dataset
from fbd_research.fertility.asfr.tft.utils import make_training_validation_sets
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import \
    optimize_hyperparameters
from scipy.special import logit

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# this contains all data, even cells not yet forecasted.
past_start = 1970
forecast_start = 2022
forecast_end = 2100

n_val_years = 10
n_train_years = forecast_start - past_start - n_val_years

df = get_dataset(past_start, forecast_start, forecast_end)

# a little bit of data pruning
df["asfr"] = logit(df["asfr"])  # we model in logit space
df = df.fillna(0)  # because TFT does not like NaNs

past_df = df.query(f"year_id < {forecast_start}")

print(len(past_df))

del df
gc.collect()

# training/validation only need past_df
training_dataset, validation_dataset, train_dataloader, val_dataloader =\
    make_training_validation_sets(past_df, n_train_years, n_val_years,
                                  num_workers=0)

# learning rate is only one of the many hyperparameters.
# create study to determine good hyperparameters.
# only need this once.
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=10,
    max_epochs=5000,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 16),
    learning_rate_range=(0.002, 0.02),
    dropout_range=(0.1, 0.5),
    trainer_kwargs=dict(limit_train_batches=10000),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=True,
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print("")
print("best parameters: ", study.best_trial.params)
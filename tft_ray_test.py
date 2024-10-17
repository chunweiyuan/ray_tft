"""A script just to see if Ray can work with TFT,
using open-source data.
"""
import lightning.pytorch as pl
import numpy as np
import ray
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import QuantileLoss

data = get_stallion_data()

# add time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# add additional features
data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")

max_prediction_length = 6
max_encoder_length = 24
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict 
# the last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True,
                                            stop_randomization=True)

# ----- below here pertains to Ray ----- #

n_tasks = 4  # number of concurrent training tasks (>1 leads to slow training)

runtime_env = {"env_vars": {"NCCL_SOCKET_IFNAME": "lo,docker0"}}

ray.init(include_dashboard=False,
         num_cpus=n_tasks + 1,  # just some number >= n_tasks
         object_store_memory=2 * 1e9,  # large enough
         runtime_env=runtime_env)


@ray.remote(num_cpus=1)
def train_task(seed: int,
               training: TimeSeriesDataSet,
               validation: TimeSeriesDataSet) -> TemporalFusionTransformer:
    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True,
                                              batch_size=batch_size,
                                              num_workers=0)
    val_dataloader = validation.to_dataloader(train=False,
                                              batch_size=batch_size * 10,
                                              num_workers=0)

    pl.seed_everything(seed)

    trainer = pl.Trainer(
        accelerator="cpu",
        gradient_clip_val=0.1,
        max_epochs=1,  # just so that it runs for a while
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="Ranger",
        reduce_on_plateau_patience=4)

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    trainer.fit(tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    return tft


tfts = []

for seed in range(n_tasks):
    tft = train_task.remote(seed, training, validation)
    tfts.append(tft)


tfts = ray.get(tfts)

ray.shutdown()

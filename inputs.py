"""All functions pertaining to I/O or transformations of inputs.
"""
import pandas as pd


def get_dataset(past_start: int,
                forecast_start: int,
                forecast_end: int) -> tuple[pd.DataFrame, int]:
    """Get outcome and covariates into a dataframe.

    The function captures all data ingestion logics.

    Index columns are location_id and age.
    Returns years past_start through forecast_end.

    Args:
        past_start (int): first past year.
        forecast_start (int): first forecast year.
        forecast_end (int): last forecast year.

    Returns:
        (pd.DataFrame): DataFrame that includes past asfr,
            along with past/future covariates.  Also has years for
            past_start, forecast_start, forecast_end.
    """
    # redacted for security reasons
    df = pd.DataFrame()

    return df

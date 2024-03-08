import polars as pl

from ebrec.utils._constants import (
    DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
)


def min_max_impression_time_history(
    df: pl.DataFrame, timestamp_col: str = DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL
):
    """
    Check min/max for user history timestamp column.
    """
    return (
        df.select(pl.col(timestamp_col))
        .with_columns(
            pl.col(timestamp_col).list.eval(pl.element().min()).explode().alias("min")
        )
        .with_columns(
            pl.col(timestamp_col).list.eval(pl.element().max()).explode().alias("max")
        )
        .select(pl.col("min").min(), pl.col("max").max())
    )


def min_max_impression_time_behaviors(
    df: pl.DataFrame, timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL
):
    """
    Check min/max for behaviors timestamp column.
    """
    return df.select(
        pl.col(timestamp_col).min().alias("min"),
        pl.col(timestamp_col).max().alias("max"),
    )

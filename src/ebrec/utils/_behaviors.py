from typing import Any, Iterable
from pathlib import Path
from tqdm import tqdm
import warnings
import datetime
import inspect


from ebrec.utils._polars import (
    slice_join_dataframes,
    _check_columns_in_df,
    drop_nulls_from_list,
    generate_unique_name,
    shuffle_list_column,
)
import polars as pl

from ebrec.utils._constants import *
from ebrec.utils._python import create_lookup_dict


def create_binary_labels_column(
    df: pl.DataFrame,
    shuffle: bool = False,
    seed: int = None,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    label_col: str = DEFAULT_LABELS_COL,
) -> pl.DataFrame:
    """Creates a new column in a DataFrame containing binary labels indicating
    whether each article ID in the "article_ids" column is present in the corresponding
    "list_destination" column.

    Args:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: A new DataFrame with an additional "labels" column.

    Examples:
    >>> from ebrec.utils._constants import (
            DEFAULT_CLICKED_ARTICLES_COL,
            DEFAULT_INVIEW_ARTICLES_COL,
            DEFAULT_LABELS_COL,
        )
    >>> df = pl.DataFrame(
            {
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [4, 5, 6], [7, 8]],
                DEFAULT_CLICKED_ARTICLES_COL: [[2, 3, 4], [3, 5], None],
            }
        )
    >>> create_binary_labels_column(df)
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [1, 2, 3]          ┆ [2, 3, 4]           ┆ [0, 1, 1] │
        │ [4, 5, 6]          ┆ [3, 5]              ┆ [0, 1, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    >>> create_binary_labels_column(df.lazy(), shuffle=True, seed=123).collect()
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [3, 1, 2]          ┆ [2, 3, 4]           ┆ [1, 0, 1] │
        │ [5, 6, 4]          ┆ [3, 5]              ┆ [1, 0, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    Test_:
    >>> assert create_binary_labels_column(df, shuffle=False)[DEFAULT_LABELS_COL].to_list() == [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0],
        ]
    >>> assert create_binary_labels_column(df, shuffle=True)[DEFAULT_LABELS_COL].list.sum().to_list() == [
            2,
            1,
            0,
        ]
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")

    df = df.with_row_index(GROUPBY_ID)

    if shuffle:
        df = shuffle_list_column(df, column=inview_col, seed=seed)

    df_labels = (
        df.explode(inview_col)
        .with_columns(
            pl.col(inview_col).is_in(pl.col(clicked_col)).cast(pl.Int8).alias(label_col)
        )
        .group_by(GROUPBY_ID)
        .agg(label_col)
    )
    return (
        df.join(df_labels, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS + [label_col])
    )


def create_user_id_to_int_mapping(
    df: pl.DataFrame, user_col: str = DEFAULT_USER_COL, value_str: str = "id"
):
    return create_lookup_dict(
        df.select(pl.col(user_col).unique()).with_row_index(value_str),
        key=user_col,
        value=value_str,
    )


def filter_minimum_negative_samples(
    df,
    n: int,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
) -> pl.DataFrame:
    """
    >>> from ebrec.utils._constants import DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [1], [1, 2, 3]],
                DEFAULT_CLICKED_ARTICLES_COL: [[1], [1], [1, 2]],
            }
        )
    >>> filter_minimum_negative_samples(df, n=1)
        shape: (2, 2)
        ┌────────────────────┬─────────────────────┐
        │ article_ids_inview ┆ article_ids_clicked │
        │ ---                ┆ ---                 │
        │ list[i64]          ┆ list[i64]           │
        ╞════════════════════╪═════════════════════╡
        │ [1, 2, 3]          ┆ [1]                 │
        │ [1, 2, 3]          ┆ [1, 2]              │
        └────────────────────┴─────────────────────┘
    >>> filter_minimum_negative_samples(df, n=2)
        shape: (3, 2)
        ┌─────────────┬──────────────────┐
        │ article_ids ┆ list_destination │
        │ ---         ┆ ---              │
        │ list[i64]   ┆ list[i64]        │
        ╞═════════════╪══════════════════╡
        │ [1, 2, 3]   ┆ [1]              │
        └─────────────┴──────────────────┘
    """
    return (
        df.filter((pl.col(inview_col).list.len() - pl.col(clicked_col).list.len()) >= n)
        if n is not None and n > 0
        else df
    )


def ebnerd_from_path(
    path: Path,
    history_size: int = 30,
    padding: int = 0,
    user_col: str = DEFAULT_USER_COL,
    history_aids_col: str = DEFAULT_HISTORY_ARTICLE_ID_COL,
) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(user_col, history_aids_col)
        .pipe(
            truncate_history,
            column=history_aids_col,
            history_size=history_size,
            padding_value=padding,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=user_col,
            how="left",
        )
    )
    return df_behaviors


def filter_read_times(df, n: int, column: str) -> pl.DataFrame:
    """
    Use this to set the cutoff for 'read_time' and 'next_read_time'
    """
    return (
        df.filter(pl.col(column) >= n)
        if column in df and n is not None and n > 0
        else df
    )


def unique_article_ids_in_behaviors(
    df: pl.DataFrame,
    col: str = "ids",
    item_col: str = DEFAULT_ARTICLE_ID_COL,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
) -> pl.Series:
    """
    Examples:
        >>> df = pl.DataFrame({
                DEFAULT_ARTICLE_ID_COL: [1, 2, 3, 4],
                DEFAULT_INVIEW_ARTICLES_COL: [[2, 3], [1, 4], [4], [1, 2, 3]],
                DEFAULT_CLICKED_ARTICLES_COL: [[], [2], [3, 4], [1]],
            })
        >>> unique_article_ids_in_behaviors(df).sort()
            [
                1
                2
                3
                4
            ]
    """
    df = df.lazy()
    return (
        pl.concat(
            (
                df.select(pl.col(item_col).unique().alias(col)),
                df.select(pl.col(inview_col).explode().unique().alias(col)),
                df.select(pl.col(clicked_col).explode().unique().alias(col)),
            )
        )
        .drop_nulls()
        .unique()
        .collect()
    ).to_series()


def add_known_user_column(
    df: pl.DataFrame,
    known_users: Iterable[int],
    user_col: str = DEFAULT_USER_COL,
    known_user_col: str = DEFAULT_KNOWN_USER_COL,
) -> pl.DataFrame:
    """
    Adds a new column to the DataFrame indicating whether the user ID is in the list of known users.
    Args:
        df: A Polars DataFrame object.
        known_users: An iterable of integers representing the known user IDs.
    Returns:
        A new Polars DataFrame with an additional column 'is_known_user' containing a boolean value
        indicating whether the user ID is in the list of known users.
    Examples:
        >>> df = pl.DataFrame({'user_id': [1, 2, 3, 4]})
        >>> add_known_user_column(df, [2, 4])
            shape: (4, 2)
            ┌─────────┬───────────────┐
            │ user_id ┆ is_known_user │
            │ ---     ┆ ---           │
            │ i64     ┆ bool          │
            ╞═════════╪═══════════════╡
            │ 1       ┆ false         │
            │ 2       ┆ true          │
            │ 3       ┆ false         │
            │ 4       ┆ true          │
            └─────────┴───────────────┘
    """
    return df.with_columns(pl.col(user_col).is_in(known_users).alias(known_user_col))


def sample_article_ids(
    df: pl.DataFrame,
    n: int,
    with_replacement: bool = False,
    seed: int = None,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Randomly sample article IDs from each row of a DataFrame with or without replacement

    Args:
        df: A polars DataFrame containing the column of article IDs to be sampled.
        n: The number of article IDs to sample from each list.
        with_replacement: A boolean indicating whether to sample with replacement.
            Default is False.
        seed: An optional seed to use for the random number generator.

    Returns:
        A new polars DataFrame with the same columns as `df`, but with the article
        IDs in the specified column replaced by a list of `n` sampled article IDs.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "clicked": [
                    [1],
                    [4, 5],
                    [7, 8, 9],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    ["A", "B", "C"],
                    ["D", "E", "F"],
                    ["G", "H", "I"],
                ],
                "col" : [
                    ["h"],
                    ["e"],
                    ["y"]
                ]
            }
        )
    >>> print(df)
        shape: (3, 3)
        ┌──────────────────┬─────────────────┬───────────┐
        │ list_destination ┆ article_ids     ┆ col       │
        │ ---              ┆ ---             ┆ ---       │
        │ list[i64]        ┆ list[str]       ┆ list[str] │
        ╞══════════════════╪═════════════════╪═══════════╡
        │ [1]              ┆ ["A", "B", "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "E", "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "H", "I"] ┆ ["y"]     │
        └──────────────────┴─────────────────┴───────────┘
    >>> sample_article_ids(df, n=2, seed=42)
        shape: (3, 3)
        ┌──────────────────┬─────────────┬───────────┐
        │ list_destination ┆ article_ids ┆ col       │
        │ ---              ┆ ---         ┆ ---       │
        │ list[i64]        ┆ list[str]   ┆ list[str] │
        ╞══════════════════╪═════════════╪═══════════╡
        │ [1]              ┆ ["A", "C"]  ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "F"]  ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "I"]  ┆ ["y"]     │
        └──────────────────┴─────────────┴───────────┘
    >>> sample_article_ids(df.lazy(), n=4, with_replacement=True, seed=42).collect()
        shape: (3, 3)
        ┌──────────────────┬───────────────────┬───────────┐
        │ list_destination ┆ article_ids       ┆ col       │
        │ ---              ┆ ---               ┆ ---       │
        │ list[i64]        ┆ list[str]         ┆ list[str] │
        ╞══════════════════╪═══════════════════╪═══════════╡
        │ [1]              ┆ ["A", "A", … "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "D", … "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "G", … "I"] ┆ ["y"]     │
        └──────────────────┴───────────────────┴───────────┘
    """
    _check_columns_in_df(df, [inview_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")
    df = df.with_row_count(name=GROUPBY_ID)

    df_ = (
        df.explode(inview_col)
        .group_by(GROUPBY_ID)
        .agg(
            pl.col(inview_col).sample(n=n, with_replacement=with_replacement, seed=seed)
        )
    )
    return (
        df.drop(inview_col)
        .join(df_, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS)
    )


def remove_positives_from_inview(
    df: pl.DataFrame,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
):
    """Removes all positive article IDs from a DataFrame column containing inview articles and another column containing
    clicked articles. Only negative article IDs (i.e., those that appear in the inview articles column but not in the
    clicked articles column) are retained.

    Args:
        df (pl.DataFrame): A DataFrame with columns containing inview articles and clicked articles.

    Returns:
        pl.DataFrame: A new DataFrame with only negative article IDs retained.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                DEFAULT_CLICKED_ARTICLES_COL: [
                    [1, 2],
                    [1],
                    [3],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ],
            }
        )
    >>> remove_positives_from_inview(df)
        shape: (3, 3)
        ┌─────────┬─────────────────────┬────────────────────┐
        │ user_id ┆ article_ids_clicked ┆ article_ids_inview │
        │ ---     ┆ ---                 ┆ ---                │
        │ i64     ┆ list[i64]           ┆ list[i64]          │
        ╞═════════╪═════════════════════╪════════════════════╡
        │ 1       ┆ [1, 2]              ┆ [3]                │
        │ 1       ┆ [1]                 ┆ [2, 3]             │
        │ 2       ┆ [3]                 ┆ [1, 2]             │
        └─────────┴─────────────────────┴────────────────────┘
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    negative_article_ids = (
        list(filter(lambda x: x not in clicked, inview))
        for inview, clicked in zip(df[inview_col].to_list(), df[clicked_col].to_list())
    )
    return df.with_columns(pl.Series(inview_col, list(negative_article_ids)))


def sampling_strategy_wu2019(
    df: pl.DataFrame,
    npratio: int,
    shuffle: bool = False,
    with_replacement: bool = True,
    seed: int = None,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
    clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Samples negative articles from the inview article pool for a given negative-position-ratio (npratio).
    The npratio (negative article per positive article) is defined as the number of negative article samples
    to draw for each positive article sample.

    This function follows the sampling strategy introduced in the paper "NPA: Neural News Recommendation with
    Personalized Attention" by Wu et al. (KDD '19).

    This is done according to the following steps:
    1. Remove the positive click-article id pairs from the DataFrame.
    2. Explode the DataFrame based on the clicked articles column.
    3. Downsample the inview negative article ids for each exploded row using the specified npratio, either
        with or without replacement.
    4. Concatenate the clicked articles back to the inview articles as lists.
    5. Convert clicked articles column to type List(Int)

    References:
        Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. 2019.
        Npa: Neural news recommendation with personalized attention. In KDD, pages 2576-2584. ACM.

    Args:
        df (pl.DataFrame): The input DataFrame containing click-article id pairs.
        npratio (int): The ratio of negative in-view article ids to positive click-article ids.
        shuffle (bool, optional): Whether to shuffle the order of the in-view article ids in each list. Default is True.
        with_replacement (bool, optional): Whether to sample the inview article ids with or without replacement.
            Default is True.
        seed (int, optional): Random seed for reproducibility. Default is None.
        inview_col (int, optional): inview column name. Default is DEFAULT_INVIEW_ARTICLES_COL,
        clicked_col (int, optional): clicked column name. Default is DEFAULT_CLICKED_ARTICLES_COL,

    Returns:
        pl.DataFrame: A new DataFrame with downsampled in-view article ids for each click according to the specified npratio.
        The DataFrame has the same columns as the input DataFrame.

    Raises:
        ValueError: If npratio is less than 0.
        ValueError: If the input DataFrame does not contain the necessary columns.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL
    >>> import polars as pl
    >>> df = pl.DataFrame(
            {
                "impression_id": [0, 1, 2, 3],
                "user_id": [1, 1, 2, 3],
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3], [1]],
                DEFAULT_CLICKED_ARTICLES_COL: [[1, 2], [1, 3], [1], [1]],
            }
        )
    >>> df
        shape: (4, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [1, 2, 3]          ┆ [1, 2]              │
        │ 1             ┆ 1       ┆ [1, 2, … 4]        ┆ [1, 3]              │
        │ 2             ┆ 2       ┆ [1, 2, 3]          ┆ [1]                 │
        │ 3             ┆ 3       ┆ [1]                ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]             ┆ [1]                 │
        │ 0             ┆ 1       ┆ [3, 2]             ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 1]             ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 3]             ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 1]             ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, 1]          ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=True, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]             ┆ [1]                 │
        │ 0             ┆ 1       ┆ [2, 3]             ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 1]             ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 3]             ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 1]             ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, 1]          ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=2, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 3, 1]          ┆ [1]                 │
        │ 0             ┆ 1       ┆ [3, 3, 2]          ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 2, 1]          ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 2, 3]          ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 2, 1]          ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, null, 1]    ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    # If we use without replacement, we need to ensure there are enough negative samples:
    >>> sampling_strategy_wu2019(df, npratio=2, shuffle=False, with_replacement=False, seed=123)
        polars.exceptions.ShapeError: cannot take a larger sample than the total population when `with_replacement=false`
    ## Either you'll have to remove the samples or split the dataframe yourself and only upsample the samples that doesn't have enough
    >>> min_neg = 2
    >>> sampling_strategy_wu2019(
            df.filter(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len() > (min_neg + 1)),
            npratio=min_neg,
            shuffle=False,
            with_replacement=False,
            seed=123,
        )
        shape: (2, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ i64                 │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 1             ┆ 1       ┆ [2, 4, 1]          ┆ 1                   │
        │ 1             ┆ 1       ┆ [2, 4, 3]          ┆ 3                   │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    """
    df = (
        # Step 1: Remove the positive 'article_id' from inview articles
        df.pipe(
            remove_positives_from_inview, inview_col=inview_col, clicked_col=clicked_col
        )
        # Step 2: Explode the DataFrame based on the clicked articles column
        .explode(clicked_col)
        # Step 3: Downsample the inview negative 'article_id' according to npratio (negative 'article_id' per positive 'article_id')
        .pipe(
            sample_article_ids,
            n=npratio,
            with_replacement=with_replacement,
            seed=seed,
            inview_col=inview_col,
        )
        # Step 4: Concatenate the clicked articles back to the inview articles as lists
        .with_columns(pl.concat_list([inview_col, clicked_col]))
        # Step 5: Convert clicked articles column to type List(Int):
        .with_columns(pl.col(inview_col).list.tail(1).alias(clicked_col))
    )
    if shuffle:
        df = shuffle_list_column(df, inview_col, seed)
    return df


def truncate_history(
    df: pl.DataFrame,
    column: str,
    history_size: int,
    padding_value: Any = None,
    enable_warning: bool = True,
) -> pl.DataFrame:
    """Truncates the history of a column containing a list of items.

    It is the tail of the values, i.e. the history ids should ascending order
    because each subsequent element (original timestamp) is greater than the previous element

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to truncate.
        history_size (int): The maximum size of the history to retain.
        padding_value (Any): Pad each list with specified value, ensuring
            equal length to each element. Default is None (no padding).
        enable_warning (bool): warn the user that history is expected in ascedings order.
            Default is True

    Returns:
        pl.DataFrame: A new DataFrame with the specified column truncated.

    Examples:
    >>> df = pl.DataFrame(
            {"id": [1, 2, 3], "history": [["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]]}
        )
    >>> df
        shape: (3, 2)
        ┌─────┬───────────────────┐
        │ id  ┆ history           │
        │ --- ┆ ---               │
        │ i64 ┆ list[str]         │
        ╞═════╪═══════════════════╡
        │ 1   ┆ ["a", "b", "c"]   │
        │ 2   ┆ ["d", "e", … "g"] │
        │ 3   ┆ ["h", "i"]        │
        └─────┴───────────────────┘
    >>> truncate_history(df, 'history', 3)
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["h", "i"]      │
        └─────┴─────────────────┘
    >>> truncate_history(df.lazy(), 'history', 3, '-').collect()
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["-", "h", "i"] │
        └─────┴─────────────────┘
    """
    if enable_warning:
        function_name = inspect.currentframe().f_code.co_name
        warnings.warn(f"{function_name}: The history IDs expeced in ascending order")
    if padding_value is not None:
        df = df.with_columns(
            pl.col(column)
            .list.reverse()
            .list.eval(pl.element().extend_constant(padding_value, n=history_size))
            .list.reverse()
        )
    return df.with_columns(pl.col(column).list.tail(history_size))


def create_dynamic_history(
    df: pl.DataFrame,
    history_size: int,
    history_col: str = "history_dynamic",
    user_col: str = DEFAULT_USER_COL,
    item_col: str = DEFAULT_ARTICLE_ID_COL,
    timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL,
) -> pl.DataFrame:
    """Generates a dynamic history of user interactions with articles based on a given DataFrame.

    Beaware, the groupby_rolling will add all the Null values, which can only be removed afterwards.
    Unlike the 'create_fixed_history' where we first remove all the Nulls, we can only do this afterwards.
    As a results, the 'history_size' might be set to N but after removal of Nulls it is (N-n_nulls) long.

    Args:
        df (pl.DataFrame): A Polars DataFrame with columns 'user_id', 'article_id', and 'first_page_time'.
        history_size (int): The maximum number of previous interactions to include in the dynamic history for each user.

    Returns:
        pl.DataFrame: A new Polars DataFrame with the same columns as the input DataFrame, plus two new columns per user:
        - 'dynamic_article_id': a list of up to 'history_size' article IDs representing the user's previous interactions,
            ordered from most to least recent. If there are fewer than 'history_size' previous interactions, the list
            is padded with 'None' values.
    Raises:
        ValueError: If the input DataFrame does not contain columns 'user_id', 'article_id', and 'first_page_time'.

    Examples:
    >>> from ebrec.utils._constants import (
            DEFAULT_IMPRESSION_TIMESTAMP_COL,
            DEFAULT_ARTICLE_ID_COL,
            DEFAULT_USER_COL,
        )
    >>> df = pl.DataFrame(
            {
                DEFAULT_USER_COL: [0, 0, 0, 1, 1, 1, 0, 2],
                DEFAULT_ARTICLE_ID_COL: [
                    9604210,
                    9634540,
                    9640420,
                    9647983,
                    9647984,
                    9647981,
                    None,
                    None,
                ],
                DEFAULT_IMPRESSION_TIMESTAMP_COL: [
                    datetime.datetime(2023, 2, 18),
                    datetime.datetime(2023, 2, 18),
                    datetime.datetime(2023, 2, 25),
                    datetime.datetime(2023, 2, 22),
                    datetime.datetime(2023, 2, 21),
                    datetime.datetime(2023, 2, 23),
                    datetime.datetime(2023, 2, 19),
                    datetime.datetime(2023, 2, 26),
                ],
            }
        )
    >>> create_dynamic_history(df, 3)
        shape: (8, 4)
        ┌─────────┬────────────┬─────────────────────┬────────────────────┐
        │ user_id ┆ article_id ┆ impression_time     ┆ history_dynamic    │
        │ ---     ┆ ---        ┆ ---                 ┆ ---                │
        │ i64     ┆ i64        ┆ datetime[μs]        ┆ list[i64]          │
        ╞═════════╪════════════╪═════════════════════╪════════════════════╡
        │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ []                 │
        │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ [9604210]          │
        │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ [9604210, 9634540] │
        │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ [9604210, 9634540] │
        │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ []                 │
        │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ [9647984]          │
        │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ [9647984, 9647983] │
        │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ []                 │
        └─────────┴────────────┴─────────────────────┴────────────────────┘
    """
    _check_columns_in_df(df, [user_col, timestamp_col, item_col])
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    df = df.sort([user_col, timestamp_col])
    return (
        df.with_columns(
            # DYNAMIC HISTORY START
            df.with_row_index(name=GROUPBY_ID)
            .with_columns(pl.col([GROUPBY_ID]).cast(pl.Int64))
            .rolling(
                index_column=GROUPBY_ID,
                period=f"{history_size}i",
                closed="left",
                by=[user_col],
            )
            .agg(pl.col(item_col).alias(history_col))
            # DYNAMIC HISTORY END
        )
        .pipe(drop_nulls_from_list, column=history_col)
        .drop(GROUPBY_ID)
    )


def create_fixed_history(
    df: pl.DataFrame,
    dt_cutoff: datetime,
    history_size: int = None,
    history_col: str = "history_fixed",
    user_col: str = DEFAULT_USER_COL,
    item_col: str = DEFAULT_ARTICLE_ID_COL,
    timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL,
) -> pl.DataFrame:
    """
    Create fixed histories for each user in a dataframe of user browsing behavior.

    Args:
        df (pl.DataFrame): A dataframe with columns "user_id", "first_page_time", and "article_id", representing user browsing behavior.
        dt_cutoff (datetime): A datetime object representing the cutoff time. Only browsing behavior before this time will be considered.
        history_size (int, optional): The maximum number of previous interactions to include in the fixed history for each user (using tail). Default is None.
            If None, all interactions are included.

    Returns:
        pl.DataFrame: A modified dataframe with columns "user_id" and "fixed_article_id". Each row represents a user and their fixed browsing history,
        which is a list of article IDs. The "fixed_" prefix is added to distinguish the fixed history from the original "article_id" column.

    Raises:
        ValueError: If the input dataframe does not contain the required columns.

    Examples:
        >>> from ebrec.utils._constants import (
                DEFAULT_IMPRESSION_TIMESTAMP_COL,
                DEFAULT_ARTICLE_ID_COL,
                DEFAULT_USER_COL,
            )
        >>> df = pl.DataFrame(
                {
                    DEFAULT_USER_COL: [0, 0, 0, 1, 1, 1, 0, 2],
                    DEFAULT_ARTICLE_ID_COL: [
                        9604210,
                        9634540,
                        9640420,
                        9647983,
                        9647984,
                        9647981,
                        None,
                        None,
                    ],
                    DEFAULT_IMPRESSION_TIMESTAMP_COL: [
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 25),
                        datetime.datetime(2023, 2, 22),
                        datetime.datetime(2023, 2, 21),
                        datetime.datetime(2023, 2, 23),
                        datetime.datetime(2023, 2, 19),
                        datetime.datetime(2023, 2, 26),
                    ],
                }
            )
        >>> dt_cutoff = datetime.datetime(2023, 2, 24)
        >>> create_fixed_history(df.lazy(), dt_cutoff).collect()
            shape: (8, 4)
            ┌─────────┬────────────┬─────────────────────┬─────────────────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ history_fixed               │
            │ ---     ┆ ---        ┆ ---                 ┆ ---                         │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ list[i64]                   │
            ╞═════════╪════════════╪═════════════════════╪═════════════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ [9604210, 9634540]          │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ [9604210, 9634540]          │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ [9604210, 9634540]          │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ [9604210, 9634540]          │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ [9647984, 9647983, 9647981] │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ [9647984, 9647983, 9647981] │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ [9647984, 9647983, 9647981] │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ null                        │
            └─────────┴────────────┴─────────────────────┴─────────────────────────────┘
        >>> create_fixed_history(df.lazy(), dt_cutoff, 1).collect()
            shape: (8, 4)
            ┌─────────┬────────────┬─────────────────────┬───────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ history_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---           │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ list[i64]     │
            ╞═════════╪════════════╪═════════════════════╪═══════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ [9634540]     │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ [9634540]     │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ [9634540]     │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ [9634540]     │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ [9647981]     │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ [9647981]     │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ [9647981]     │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ null          │
            └─────────┴────────────┴─────────────────────┴───────────────┘
    """
    _check_columns_in_df(df, [user_col, timestamp_col, item_col])

    df = df.sort(user_col, timestamp_col)
    df_history = (
        df.select(user_col, timestamp_col, item_col)
        .filter(pl.col(item_col).is_not_null())
        .filter(pl.col(timestamp_col) < dt_cutoff)
        .group_by(user_col)
        .agg(
            pl.col(item_col).alias(history_col),
        )
    )
    if history_size is not None:
        df_history = df_history.with_columns(
            pl.col(history_col).list.tail(history_size)
        )
    return df.join(df_history, on=user_col, how="left")


def create_fixed_history_aggr_columns(
    df: pl.DataFrame,
    dt_cutoff: datetime,
    history_size: int = None,
    columns: list[str] = [],
    suffix: str = "_fixed",
    user_col: str = DEFAULT_USER_COL,
    item_col: str = DEFAULT_ARTICLE_ID_COL,
    timestamp_col: str = DEFAULT_IMPRESSION_TIMESTAMP_COL,
) -> pl.DataFrame:
    """
    This function aggregates historical data in a Polars DataFrame based on a specified cutoff datetime and user-defined columns.
    The historical data is fixed to a given number of most recent records per user.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame OR LazyFrame.
        dt_cutoff (datetime): The cutoff datetime for filtering the history.
        history_size (int, optional): The number of most recent records to keep for each user.
            If None, all history before the cutoff is kept.
        columns (list[str], optional): List of column names to be included in the aggregation.
            These columns are in addition to the mandatory 'user_id', 'article_id', and 'impression_timestamp'.
        lazy_output (bool, optional): whether to output df as LazyFrame.

    Returns:
        pl.DataFrame: A new DataFrame with the original columns and added columns for each specified column in the history.
        Each new column contains a list of historical values.

    Raises:
        ValueError: If the input dataframe does not contain the required columns.

    Examples:
        >>> from ebrec.utils._constants import (
                DEFAULT_IMPRESSION_TIMESTAMP_COL,
                DEFAULT_ARTICLE_ID_COL,
                DEFAULT_READ_TIME_COL,
                DEFAULT_USER_COL,
            )
        >>> df = pl.DataFrame(
                {
                    DEFAULT_USER_COL: [0, 0, 0, 1, 1, 1, 0, 2],
                    DEFAULT_ARTICLE_ID_COL: [
                        9604210,
                        9634540,
                        9640420,
                        9647983,
                        9647984,
                        9647981,
                        None,
                        None,
                    ],
                    DEFAULT_IMPRESSION_TIMESTAMP_COL: [
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 25),
                        datetime.datetime(2023, 2, 22),
                        datetime.datetime(2023, 2, 21),
                        datetime.datetime(2023, 2, 23),
                        datetime.datetime(2023, 2, 19),
                        datetime.datetime(2023, 2, 26),
                    ],
                    DEFAULT_READ_TIME_COL: [
                        0,
                        2,
                        8,
                        13,
                        1,
                        1,
                        6,
                        1
                    ],
                    "nothing": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                }
            )
        >>> dt_cutoff = datetime.datetime(2023, 2, 24)
        >>> columns = [DEFAULT_IMPRESSION_TIMESTAMP_COL, DEFAULT_READ_TIME_COL]
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, columns=columns).collect()
            shape: (8, 8)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬─────────────────┬─────────────────────────────┬───────────────────────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ read_time_fixed ┆ article_id_fixed            ┆ impression_time_fixed             │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---             ┆ ---                         ┆ ---                               │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]       ┆ list[i64]                   ┆ list[datetime[μs]]                │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪═════════════════╪═════════════════════════════╪═══════════════════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8         ┆ null    ┆ [0, 2]          ┆ [9604210, 9634540]          ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1         ┆ null    ┆ [1, 13, 1]      ┆ [9647984, 9647983, 9647981] ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13        ┆ null    ┆ [1, 13, 1]      ┆ [9647984, 9647983, 9647981] ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1         ┆ null    ┆ [1, 13, 1]      ┆ [9647984, 9647983, 9647981] ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1         ┆ null    ┆ null            ┆ null                        ┆ null                              │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴─────────────────┴─────────────────────────────┴───────────────────────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1, columns=columns).collect()
            shape: (8, 8)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬─────────────────┬──────────────────┬───────────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ read_time_fixed ┆ article_id_fixed ┆ impression_time_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---             ┆ ---              ┆ ---                   │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]       ┆ list[i64]        ┆ list[datetime[μs]]    │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪═════════════════╪══════════════════╪═══════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8         ┆ null    ┆ [2]             ┆ [9634540]        ┆ [2023-02-18 00:00:00] │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1         ┆ null    ┆ [1]             ┆ [9647981]        ┆ [2023-02-23 00:00:00] │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13        ┆ null    ┆ [1]             ┆ [9647981]        ┆ [2023-02-23 00:00:00] │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1         ┆ null    ┆ [1]             ┆ [9647981]        ┆ [2023-02-23 00:00:00] │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1         ┆ null    ┆ null            ┆ null             ┆ null                  │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴─────────────────┴──────────────────┴───────────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1).collect()
            shape: (8, 6)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬──────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ article_id_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---              │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]        │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪══════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [9634540]        │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2         ┆ null    ┆ [9634540]        │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6         ┆ null    ┆ [9634540]        │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8         ┆ null    ┆ [9634540]        │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1         ┆ null    ┆ [9647981]        │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13        ┆ null    ┆ [9647981]        │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1         ┆ null    ┆ [9647981]        │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1         ┆ null    ┆ null             │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴──────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1).head(1).collect()
            shape: (1, 6)
            ┌─────────┬────────────┬─────────────────────┬───────────┬─────────┬──────────────────┐
            │ user_id ┆ article_id ┆ impression_time     ┆ read_time ┆ nothing ┆ article_id_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---       ┆ ---     ┆ ---              │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64       ┆ null    ┆ list[i64]        │
            ╞═════════╪════════════╪═════════════════════╪═══════════╪═════════╪══════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0         ┆ null    ┆ [9634540]        │
            └─────────┴────────────┴─────────────────────┴───────────┴─────────┴──────────────────┘
    """
    _check_columns_in_df(df, [user_col, item_col, timestamp_col] + columns)
    aggr_columns = list(set([item_col] + columns))
    df = df.sort(user_col, timestamp_col)
    df_history = (
        df.select(pl.all())
        .filter(pl.col(item_col).is_not_null())
        .filter(pl.col(timestamp_col) < dt_cutoff)
        .group_by(user_col)
        .agg(
            pl.col(aggr_columns).suffix(suffix),
        )
    )
    if history_size is not None:
        for col in aggr_columns:
            df_history = df_history.with_columns(
                pl.col(col + suffix).list.tail(history_size)
            )
    return df.join(df_history, on="user_id", how="left")


def add_prediction_scores(
    df: pl.DataFrame,
    scores: Iterable[float],
    prediction_scores_col: str = "scores",
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
) -> pl.DataFrame:
    """
    Adds prediction scores to a DataFrame for the corresponding test predictions.

    Args:
        df (pl.DataFrame): The DataFrame to which the prediction scores will be added.
        test_prediction (Iterable[float]): A list, array or simialr of prediction scores for the test data.

    Returns:
        pl.DataFrame: The DataFrame with the prediction scores added.

    Raises:
        ValueError: If there is a mismatch in the lengths of the list columns.

    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "id": [1,2],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [4, 5],
                ],
            }
        )
    >>> test_prediction = [[0.3], [0.4], [0.5], [0.6], [0.7]]
    >>> add_prediction_scores(df.lazy(), test_prediction).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    ## The input can can also be an np.array
    >>> add_prediction_scores(df.lazy(), np.array(test_prediction)).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    """
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    # df_preds = pl.DataFrame()
    scores = (
        df.lazy()
        .select(pl.col(inview_col))
        .with_row_index(GROUPBY_ID)
        .explode(inview_col)
        .with_columns(pl.Series(prediction_scores_col, scores).explode())
        .group_by(GROUPBY_ID)
        .agg(inview_col, prediction_scores_col)
        .sort(GROUPBY_ID)
        .collect()
    )
    return df.with_columns(scores.select(prediction_scores_col)).drop(GROUPBY_ID)


def down_sample_on_users(
    df: pl.DataFrame,
    n: int,
    user_col: str = DEFAULT_USER_COL,
    seed: int = None,
) -> pl.DataFrame:
    """
    Down-samples a DataFrame by randomly selecting up to 'n' rows per unique user.

    Args:
        df (pl.DataFrame): The input DataFrame to be down-sampled.
        n (int): The maximum number of rows to retain per user.
        user_col (str): The column representing user identifiers. Defaults to DEFAULT_USER_COL.
        seed (int, optional): The random seed for reproducibility. Defaults to None.

    Returns:
        pl.DataFrame: A down-sampled DataFrame with at most 'n' rows per user.
    >>> import polars as pl
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 3],
                "value": [10, 20, 30, 40, 50, 60],
            }
        )
    >>> down_sample_on_users(df, n=2, user_col="user_id", seed=42)
        shape: (5, 2)
        ┌─────────┬───────┐
        │ user_id ┆ value │
        │ ---     ┆ ---   │
        │ i64     ┆ i64   │
        ╞═════════╪═══════╡
        │ 1       ┆ 10    │
        │ 1       ┆ 20    │
        │ 2       ┆ 40    │
        │ 2       ┆ 50    │
        │ 3       ┆ 60    │
        └─────────┴───────┘
    """

    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    df = df.with_row_index(GROUPBY_ID)

    filter_index = (
        df.sample(fraction=1.0, shuffle=True, seed=seed)
        .group_by(pl.col(user_col))
        .agg(GROUPBY_ID)
        .with_columns(pl.col(GROUPBY_ID).list.tail(n))
    ).select(pl.col(GROUPBY_ID).explode())

    return df.filter(pl.col(GROUPBY_ID).is_in(filter_index)).drop(GROUPBY_ID)

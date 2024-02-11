from typing import Any, Iterable
from tqdm import tqdm
import polars as pl
import datetime

from newssources.utils_polars import (
    _check_columns_in_df,
    drop_nulls_from_list,
    shuffle_list_column,
    split_df_in_n,
)

from newssources.constants import (
    IMPRESSION_TIMESTAMP,
    CLICKED_ARTICLES,
    INVIEW_ARTICLES,
    DYNAMIC_HISTORY,
    FIXED_HISTORY,
    FIXED_SUFFIX,
    IS_SSO_USER,
    ARTICLE_ID,
    USER_ID,
    LABELS,
)
from newssources.config import TIMEIT

from newssources.utils import generate_unique_name


def create_binary_labels_column(
    df: pl.DataFrame, shuffle: bool = False, seed: int = None
) -> pl.DataFrame:
    """Creates a new column in a DataFrame containing binary labels indicating
    whether each article ID in the "article_ids" column is present in the corresponding
    "list_destination" column.

    Args:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: A new DataFrame with an additional "labels" column.

    Example:
    >>> from newssources.constants import INVIEW_ARTICLES, CLICKED_ARTICLES
    >>> df = pl.DataFrame(
            {
                INVIEW_ARTICLES: [[1, 2, 3], [3, 4, 5], [1, 2]],
                CLICKED_ARTICLES: [[2, 3, 4], [3, 5], None],
            }
        )
    >>> create_binary_labels_column(df)
        shape: (3, 3)
        ┌─────────────┬──────────────────┬───────────┐
        │ article_ids ┆ list_destination ┆ labels    │
        │ ---         ┆ ---              ┆ ---       │
        │ list[i64]   ┆ list[i64]        ┆ list[i8]  │
        ╞═════════════╪══════════════════╪═══════════╡
        │ [1, 2, 3]   ┆ [2, 3, 4]        ┆ [0, 1, 1] │
        │ [3, 4, 5]   ┆ [3, 5]           ┆ [1, 0, 1] │
        │ [1, 2]      ┆ null             ┆ [0, 0]    │
        └─────────────┴──────────────────┴───────────┘
    >>> create_binary_labels_column(df.lazy(), shuffle=True, seed=123).collect()
        shape: (3, 3)
        ┌─────────────┬──────────────────┬───────────┐
        │ article_ids ┆ list_destination ┆ labels    │
        │ ---         ┆ ---              ┆ ---       │
        │ list[i64]   ┆ list[i64]        ┆ list[i8]  │
        ╞═════════════╪══════════════════╪═══════════╡
        │ [3, 1, 2]   ┆ [2, 3, 4]        ┆ [1, 0, 1] │
        │ [4, 5, 3]   ┆ [3, 5]           ┆ [0, 1, 1] │
        │ [1, 2]      ┆ null             ┆ [0, 0]    │
        └─────────────┴──────────────────┴───────────┘

    Test_:
    >>> assert create_binary_labels_column(df, shuffle=False)[LABELS].to_list() == [
            [0, 1, 1],
            [1, 0, 1],
            [0, 0],
        ]
    >>> assert create_binary_labels_column(df, shuffle=True)[LABELS].list.sum().to_list() == [
            2,
            2,
            0,
        ]
    """
    _check_columns_in_df(df, [INVIEW_ARTICLES, CLICKED_ARTICLES])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")

    df = df.with_row_count(GROUPBY_ID)

    if shuffle:
        df = shuffle_list_column(df, column=INVIEW_ARTICLES, seed=seed)

    df_labels = (
        df.explode(INVIEW_ARTICLES)
        .with_columns(
            pl.col(INVIEW_ARTICLES)
            .is_in(pl.col(CLICKED_ARTICLES))
            .cast(pl.Int8)
            .alias(LABELS)
        )
        .group_by(GROUPBY_ID)
        .agg(LABELS)
    )
    return (
        df.join(df_labels, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS + [LABELS])
    )


def filter_more_inview_than_clicked(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filters the DataFrame rows where the number of inview articles is greater than the number of clicked articles.

    Args:
        df (pl.DataFrame): The input DataFrame containing columns for inview articles and clicked articles.

    Returns:
        pl.DataFrame: A new DataFrame containing only rows where the number of inview articles is greater than
        the number of clicked articles.

    Example:
        >>> from newssources.constants import INVIEW_ARTICLES, CLICKED_ARTICLES
        >>> df = pl.DataFrame({
                INVIEW_ARTICLES: [["a", "b", "c"], ["d", "e"], ["f", "g"]],
                CLICKED_ARTICLES: [["a"], ["d"], ["f", "g"]],
            })
        >>> filter_more_inview_than_clicked(df)
            shape: (2, 2)
            ┌─────────────────┬──────────────────┐
            │ article_ids     ┆ list_destination │
            │ ---             ┆ ---              │
            │ list[str]       ┆ list[str]        │
            ╞═════════════════╪══════════════════╡
            │ ["a", "b", "c"] ┆ ["a"]            │
            │ ["d", "e"]      ┆ ["d"]            │
            └─────────────────┴──────────────────┘
    """
    return df.filter(
        pl.col(INVIEW_ARTICLES).list.len() > pl.col(CLICKED_ARTICLES).list.len()
    )


def filter_is_sso_user(df) -> pl.DataFrame:
    return df.filter(IS_SSO_USER)


def create_user_id_mapping(
    df: pl.DataFrame,
    user_col: str = DEFAULT_USER_COL,
):
    return create_lookup_dict(
        df.select(pl.col(user_col).unique()).with_row_count("id"),
        key=user_col,
        value="id",
    )


def filter_minimum_negative_samples(df, n: int) -> pl.DataFrame:
    """
    >>> df = pl.DataFrame(
            {
                "article_ids": [[1, 2, 3], [1], [1, 2, 3]],
                "list_destination": [[1], [1], [1, 2]],
            }
        )
    >>> filter_minimum_negative_samples(df, n=1)
        shape: (3, 2)
        ┌─────────────┬──────────────────┐
        │ article_ids ┆ list_destination │
        │ ---         ┆ ---              │
        │ list[i64]   ┆ list[i64]        │
        ╞═════════════╪══════════════════╡
        │ [1, 2, 3]   ┆ [1]              │
        │ [1]         ┆ [1]              │
        │ [1, 2, 3]   ┆ [1, 2]           │
        └─────────────┴──────────────────┘
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
        df.filter(
            (pl.col(INVIEW_ARTICLES).list.len() - pl.col(CLICKED_ARTICLES).list.len())
            >= n
        )
        if n is not None and n > 0
        else df
    )


def filter_datetime_interval(
    df: pl.DataFrame, datetimes: list[datetime.datetime], column: str
) -> pl.DataFrame:
    """
    Example:
    >>> import polars as pl
    >>> import datetime
    >>> df = pl.DataFrame(
            {
                "first_page_time": [
                    datetime.datetime(2021, 1, day=1),
                    datetime.datetime(2021, 1, day=2),
                    datetime.datetime(2021, 1, day=3),
                    datetime.datetime(2021, 1, day=4),
                ],
                "data": [1, 2, 3, 4],
            }
        )
    >>> start_end = [datetime.datetime(2021, 1, day=1), datetime.datetime(2021, 1, day=3)]
    >>> filter_datetime_interval(df, start_end, column="first_page_time")
        shape: (2, 2)
        ┌─────────────────────┬──────┐
        │ first_page_time     ┆ data │
        │ ---                 ┆ ---  │
        │ datetime[μs]        ┆ i64  │
        ╞═════════════════════╪══════╡
        │ 2021-01-01 00:00:00 ┆ 1    │
        │ 2021-01-02 00:00:00 ┆ 2    │
        └─────────────────────┴──────┘
    >>> filter_datetime_interval(df, start_end[::-1], column="first_page_time")
        ValueError: Start datetime must be earlier than end datetime. Input: [datetime.datetime(2021, 1, 3, 0, 0), datetime.datetime(2021, 1, 1, 0, 0)]
    """
    if datetimes[0] > datetimes[1]:
        raise ValueError(
            f"Start datetime must be earlier than end datetime. Input: {datetimes}"
        )
    return df.filter(pl.col(column) >= datetimes[0]).filter(
        pl.col(column) < datetimes[1]
    )


def filter_predicable_impressions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Each impression in the dataset contains both positive and negative samples:
        inview articles and clicked articles.

    Exmaple:
    >>> df = pl.DataFrame(
            {
                "article_ids": [[1, 2, 3], None, [1, 2, 3], [1, 2]],
                "list_destination": [[1], [1], None, [2]],
                "whatever": [["h"], ["e"], ["y"], None],
            }
        )
    >>> filter_invalid_impressions(df)
        shape: (2, 3)
        ┌─────────────┬──────────────────┬───────────┐
        │ article_ids ┆ list_destination ┆ whatever  │
        │ ---         ┆ ---              ┆ ---       │
        │ list[i64]   ┆ list[i64]        ┆ list[str] │
        ╞═════════════╪══════════════════╪═══════════╡
        │ [1, 2, 3]   ┆ [1]              ┆ ["h"]     │
        │ [1, 2]      ┆ [2]              ┆ null      │
        └─────────────┴──────────────────┴───────────┘
    """
    return df.drop_nulls((INVIEW_ARTICLES, CLICKED_ARTICLES))


def filter_read_times(df, n: int, column: str) -> pl.DataFrame:
    """
    Use this to set the cutoff for 'read_time' and 'next_read_time'
    """
    return (
        df.filter(pl.col(column) >= n)
        if column in df and n is not None and n > 0
        else df
    )


def add_known_user_column(df: pl.DataFrame, known_users: Iterable[int]) -> pl.DataFrame:
    """
    Adds a new column to the DataFrame indicating whether the user ID is in the list of known users.
    Args:
        df: A Polars DataFrame object.
        known_users: An iterable of integers representing the known user IDs.
    Returns:
        A new Polars DataFrame with an additional column 'is_known_user' containing a boolean value
        indicating whether the user ID is in the list of known users.
    Example:
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
    return df.with_columns(pl.col(USER_ID).is_in(known_users).alias("is_known_user"))


def add_lagged_columns(
    df: pl.DataFrame, columns: list[str], prefix_new_column: str = "next_"
) -> pl.DataFrame:
    """
    Adds lagged columns to a Polars DataFrame based on the values in specified columns.

    Args:
        df (pl.DataFrame): The input DataFrame.
        columns (list[str]): The names of the columns to use for calculating the lagged values.
        prefix_new_column (str, optional): The prefix to use for the new column names. Defaults to "next_".

    Returns:
        pl.DataFrame: A new DataFrame with additional columns that contain the lagged values of the specified columns.

    Examples:
    >>> import polars as pl
    >>> df = pl.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "first_page_time": ["2022-01-01 01:00:00", "2022-01-01 01:30:00", "2022-01-01 01:20:00", "2022-01-02 12:50:00", "2022-01-02 12:30:00"],
            "pageviews": [10, 15, 20, 5, 10],
            "clicks": [100, 150, 200, 50, 100]
        })
    >>> add_lagged_columns(df, columns=["pageviews", "clicks"])
        shape: (5, 6)
        ┌─────────┬─────────────────────┬───────────┬────────┬────────────────┬─────────────┐
        │ user_id ┆ first_page_time     ┆ pageviews ┆ clicks ┆ next_pageviews ┆ next_clicks │
        │ ---     ┆ ---                 ┆ ---       ┆ ---    ┆ ---            ┆ ---         │
        │ i64     ┆ str                 ┆ i64       ┆ i64    ┆ i64            ┆ i64         │
        ╞═════════╪═════════════════════╪═══════════╪════════╪════════════════╪═════════════╡
        │ 1       ┆ 2022-01-01 01:00:00 ┆ 10        ┆ 100    ┆ 20             ┆ 200         │
        │ 1       ┆ 2022-01-01 01:20:00 ┆ 20        ┆ 200    ┆ 15             ┆ 150         │
        │ 1       ┆ 2022-01-01 01:30:00 ┆ 15        ┆ 150    ┆ 10             ┆ 100         │
        │ 2       ┆ 2022-01-02 12:30:00 ┆ 10        ┆ 100    ┆ 5              ┆ 50          │
        │ 2       ┆ 2022-01-02 12:50:00 ┆ 5         ┆ 50     ┆ null           ┆ null        │
        └─────────┴─────────────────────┴───────────┴────────┴────────────────┴─────────────┘
    """

    # Sanity check:
    _check_columns_in_df(df, [USER_ID, IMPRESSION_TIMESTAMP] + columns)

    df = df.sort([USER_ID, IMPRESSION_TIMESTAMP])
    # Shift lagged columns:
    new_col = df.select(pl.col(columns).prefix(prefix_new_column)).shift(-1)
    return df.with_columns(new_col)


def sample_article_ids(
    df: pl.DataFrame,
    n: int,
    with_replacement: bool = False,
    seed: int = None,
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

    Example:
    >>> df = pl.DataFrame(
            {
                "list_destination": [
                    [1],
                    [4, 5],
                    [7, 8, 9],
                ],
                "article_ids": [
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
    >>> sample_article_ids(df, n=4, with_replacement=True, seed=42)
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
    _check_columns_in_df(df, [INVIEW_ARTICLES])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")
    df = df.with_row_count(name=GROUPBY_ID)

    df_ = (
        df.lazy()
        .explode(INVIEW_ARTICLES)
        .group_by(GROUPBY_ID)
        .agg(
            pl.col(INVIEW_ARTICLES).sample(
                n=n, with_replacement=with_replacement, seed=seed
            )
        )
    )
    return (
        df.drop(INVIEW_ARTICLES)
        .join(df_.collect(), on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS)
    )


def remove_positives_from_article_ids(df: pl.DataFrame):
    """Removes all positive article IDs from a DataFrame column containing inview articles and another column containing
    clicked articles. Only negative article IDs (i.e., those that appear in the inview articles column but not in the
    clicked articles column) are retained.

    Args:
        df (pl.DataFrame): A DataFrame with columns containing inview articles and clicked articles.

    Returns:
        pl.DataFrame: A new DataFrame with only negative article IDs retained.

    Example:
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                "list_destination": [
                    [1, 2],
                    [1],
                    [3],
                ],
                "article_ids": [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ],
            }
        )
    >>> remove_positives_from_article_ids(df)
        shape: (3, 3)
        ┌─────────┬──────────────────┬─────────────┐
        │ user_id ┆ list_destination ┆ article_ids │
        │ ---     ┆ ---              ┆ ---         │
        │ i64     ┆ list[i64]        ┆ list[i64]   │
        ╞═════════╪══════════════════╪═════════════╡
        │ 1       ┆ [1, 2]           ┆ [3]         │
        │ 1       ┆ [1]              ┆ [2, 3]      │
        │ 2       ┆ [3]              ┆ [1, 2]      │
        └─────────┴──────────────────┴─────────────┘
    """
    _check_columns_in_df(df, [INVIEW_ARTICLES, CLICKED_ARTICLES])
    negative_article_ids = (
        list(filter(lambda x: x not in clicked, inview))
        for inview, clicked in zip(
            df[INVIEW_ARTICLES].to_list(), df[CLICKED_ARTICLES].to_list()
        )
    )
    return df.with_columns(pl.Series(INVIEW_ARTICLES, list(negative_article_ids)))


def sort_behaviors_on_inview_lengths(df: pl.DataFrame, num_splits: int) -> pl.DataFrame:
    """
    Sort behaviors on inview lengths and split into multiple DataFrames.

    Args:
        df (pl.DataFrame): The input DataFrame.
        num_splits (int): The number rows to based the group-ordering on.

    Returns:
        pl.DataFrame: Reordered DataFrame.

    >>> from newssources.constants import INVIEW_ARTICLES
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                INVIEW_ARTICLES: [
                    [3],
                    [1, 2],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            }
        )
    >>> sort_behaviors_on_inview_lengths(df, 2)
        shape: (4, 2)
        ┌─────────┬─────────────┐
        │ user_id ┆ article_ids │
        │ ---     ┆ ---         │
        │ i64     ┆ list[i64]   │
        ╞═════════╪═════════════╡
        │ 1       ┆ [1, 2]      │
        │ 2       ┆ [1, 1, 1]   │
        │ 1       ┆ [3]         │
        │ 2       ┆ [1, 1, 1]   │
        └─────────┴─────────────┘
    """
    # =>
    df = df.with_columns(pl.col(INVIEW_ARTICLES).list.len().alias("n")).sort("n")
    # =>
    dfs = split_df_in_n(df, num_splits)
    # =>
    true_false = [False, True] * len(dfs)
    # =>
    dfs = [
        df.sort(by="n", descending=tf).with_row_count("idx")
        for df, tf in zip(dfs, true_false)
    ]
    # =>
    df_con = pl.concat(dfs).sort(by="idx").drop("idx", "n")
    return df_con


def sampling_strategy_wu2019(
    df: pl.DataFrame,
    npratio: int,
    shuffle: bool = False,
    with_replacement: bool = True,
    seed: int = None,
) -> pl.DataFrame:
    """
    Samples negative articles from the inview article pool for a given negative-position-ratio (npratio).
    The npratio (negative article per positive article) is defined as the number of negative article samples
    to draw for each positive article sample.

    This function follows the sampling strategy introduced in the paper "NPA: Neural News Recommendation with
    Personalized Attention" by Wu et al. (KDD '19).

    This is done according to the following steps:
    1. Remove the positive click-article id pairs from the DataFrame.
    2. Add an 'impression_id' column to the DataFrame, which is used later for grouping.
    3. Explode the DataFrame based on the clicked articles column.
    4. Downsample the inview negative article ids for each exploded row using the specified npratio, either
        with or without replacement.
    5. Concatenate the clicked articles back to the inview articles as lists.
    6. Convert clicked articles column to type List(Int)

    References:
        Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. 2019.
        Npa: Neural news recommendation with personalized attention. In KDD, pages 2576-2584. ACM.

    Args:
        df (pl.DataFrame): The input DataFrame containing click-article id pairs.
        npratio (int): The ratio of negative in-view article ids to positive click-article ids.
        shuffle (bool, optional): Whether to shuffle the order of the in-view article ids in each list. Default is True.
        with_replacement (bool, optional): Whether to sample the in-view article ids with or without replacement.
            Default is True (with replacement).
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        pl.DataFrame: A new DataFrame with downsampled in-view article ids for each click according to the specified npratio.
        The DataFrame has the same columns as the input DataFrame.

    Raises:
        ValueError: If npratio is less than 0.
        ValueError: If the input DataFrame does not contain the necessary columns.

    Example:
    >>> import polars as pl
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 3],
                "article_ids": [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3], [1]],
                "list_destination": [[1, 2], [1, 3], [1], [1]],
            }
        )
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬─────────────┬──────────────────┐
        │ impression_id ┆ user_id ┆ article_ids ┆ list_destination │
        │ ---           ┆ ---     ┆ ---         ┆ ---              │
        │ u32           ┆ i64     ┆ list[i64]   ┆ list[i64]        │
        ╞═══════════════╪═════════╪═════════════╪══════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]      ┆ [1]              │
        │ 0             ┆ 1       ┆ [3, 2]      ┆ [2]              │
        │ 1             ┆ 1       ┆ [4, 1]      ┆ [1]              │
        │ 1             ┆ 1       ┆ [4, 3]      ┆ [3]              │
        │ 2             ┆ 2       ┆ [3, 1]      ┆ [1]              │
        │ 3             ┆ 3       ┆ [null, 1]   ┆ [1]              │
        └───────────────┴─────────┴─────────────┴──────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=True, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬─────────────┬──────────────────┐
        │ impression_id ┆ user_id ┆ article_ids ┆ list_destination │
        │ ---           ┆ ---     ┆ ---         ┆ ---              │
        │ u32           ┆ i64     ┆ list[i64]   ┆ list[i64]        │
        ╞═══════════════╪═════════╪═════════════╪══════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]      ┆ [1]              │
        │ 0             ┆ 1       ┆ [2, 3]      ┆ [2]              │
        │ 1             ┆ 1       ┆ [4, 1]      ┆ [1]              │
        │ 1             ┆ 1       ┆ [4, 3]      ┆ [3]              │
        │ 2             ┆ 2       ┆ [3, 1]      ┆ [1]              │
        │ 3             ┆ 3       ┆ [null, 1]   ┆ [1]              │
        └───────────────┴─────────┴─────────────┴──────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=2, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬─────────────────┬──────────────────┐
        │ impression_id ┆ user_id ┆ article_ids     ┆ list_destination │
        │ ---           ┆ ---     ┆ ---             ┆ ---              │
        │ u32           ┆ i64     ┆ list[i64]       ┆ list[i64]        │
        ╞═══════════════╪═════════╪═════════════════╪══════════════════╡
        │ 0             ┆ 1       ┆ [3, 3, 1]       ┆ [1]              │
        │ 0             ┆ 1       ┆ [3, 3, 2]       ┆ [2]              │
        │ 1             ┆ 1       ┆ [4, 2, 1]       ┆ [1]              │
        │ 1             ┆ 1       ┆ [4, 2, 3]       ┆ [3]              │
        │ 2             ┆ 2       ┆ [3, 2, 1]       ┆ [1]              │
        │ 3             ┆ 3       ┆ [null, null, 1] ┆ [1]              │
        └───────────────┴─────────┴─────────────────┴──────────────────┘
    """
    df = (
        # Step 1: Remove the positive 'article_id' from inview articles
        df.pipe(remove_positives_from_article_ids)
        # Step 2: Add 'impression_id' to the DataFrame for later grouping
        .with_row_count("impression_id")
        # Step 3: Explode the DataFrame based on the clicked articles column
        .explode(CLICKED_ARTICLES)
        # Step 4: Downsample the inview negative 'article_id' according to npratio (negative 'article_id' per positive 'article_id')
        .pipe(
            sample_article_ids, n=npratio, with_replacement=with_replacement, seed=seed
        )
        # Step 5: Concatenate the clicked articles back to the inview articles as lists
        .with_columns(pl.concat_list([INVIEW_ARTICLES, CLICKED_ARTICLES]))
        # Step 6: Convert clicked articles column to type List(Int):
        .with_columns(pl.col(INVIEW_ARTICLES).list.tail(1).alias(CLICKED_ARTICLES))
    )
    if shuffle:
        df = shuffle_list_column(df, INVIEW_ARTICLES, seed)
    return df


def truncate_history(
    df: pl.DataFrame, column: str, history_size: int, padding_value: Any = None
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

    Returns:
        pl.DataFrame: A new DataFrame with the specified column truncated.
    Example:
    >>> df = pl.DataFrame({
            'id': [1, 2, 3],
            'history': [['a', 'b', 'c'], ['d', 'e', 'f', 'g'], ['h', 'i']]
        })
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
    >>> truncate_history(df, 'history', 3, '-')
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
    if padding_value is not None:
        df = df.with_columns(
            pl.col(column)
            .list.reverse()
            .list.eval(pl.element().extend_constant(padding_value, n=history_size))
            .list.reverse()
        )
    return df.with_columns(df[column].list.tail(history_size))


def create_dynamic_history(df: pl.DataFrame, history_size: int) -> pl.DataFrame:
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
    >>> df = pl.DataFrame(
            {
                "user_id": [0, 0, 0, 1, 1, 1, 0, 2],
                "article_id": [
                    9604210,
                    9634540,
                    9640420,
                    9647983,
                    9647984,
                    9647981,
                    None,
                    None,
                ],
                "first_page_time": [
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
        │ user_id ┆ article_id ┆ first_page_time     ┆ article_id_dynamic │
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
    _check_columns_in_df(df, [USER_ID, IMPRESSION_TIMESTAMP, ARTICLE_ID])
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    df = df.sort([USER_ID, IMPRESSION_TIMESTAMP])

    return (
        df.with_columns(
            # DYNAMIC HISTORY START
            df.with_row_count(name=GROUPBY_ID)
            .with_columns(
                [
                    pl.col([GROUPBY_ID]).cast(pl.Int64),
                ]
            )
            .groupby_rolling(
                index_column=GROUPBY_ID,
                period=f"{history_size}i",
                closed="left",
                by=[USER_ID],
            )
            .agg(
                [
                    pl.col(ARTICLE_ID).alias(DYNAMIC_HISTORY),
                ]
            )
            # DYNAMIC HISTORY END
        )
        .pipe(drop_nulls_from_list, column=DYNAMIC_HISTORY)
        .drop(GROUPBY_ID)
    )


def create_fixed_history(
    df: pl.DataFrame, dt_cutoff: datetime, history_size: int = None
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
        >>> df = pl.DataFrame(
                {
                    "user_id": [0, 0, 0, 1, 1, 1, 0, 2],
                    "article_id": [
                        9604210,
                        9634540,
                        9640420,
                        9647983,
                        9647984,
                        9647981,
                        None,
                        None,
                    ],
                    "first_page_time": [
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
            │ user_id ┆ article_id ┆ first_page_time     ┆ article_id_fixed            │
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
            ┌─────────┬────────────┬─────────────────────┬──────────────────┐
            │ user_id ┆ article_id ┆ first_page_time     ┆ article_id_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---              │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ list[i64]        │
            ╞═════════╪════════════╪═════════════════════╪══════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ [9634540]        │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ [9634540]        │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ [9634540]        │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ [9634540]        │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ [9647981]        │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ [9647981]        │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ [9647981]        │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ null             │
            └─────────┴────────────┴─────────────────────┴──────────────────┘
    """
    _check_columns_in_df(df, [USER_ID, IMPRESSION_TIMESTAMP, ARTICLE_ID])

    df = df.sort(USER_ID, IMPRESSION_TIMESTAMP)
    df_history = (
        df.select(USER_ID, IMPRESSION_TIMESTAMP, ARTICLE_ID)
        .filter(pl.col(ARTICLE_ID).is_not_null())
        .filter(pl.col(IMPRESSION_TIMESTAMP) < dt_cutoff)
        .group_by(USER_ID)
        .agg(
            pl.col(ARTICLE_ID).alias(FIXED_HISTORY),
        )
    )
    if history_size is not None:
        df_history = df_history.with_columns(
            pl.col(FIXED_HISTORY).list.tail(history_size)
        )
    return df.join(df_history, on="user_id", how="left")


def create_fixed_history_aggr_columns(
    df: pl.DataFrame,
    dt_cutoff: datetime,
    history_size: int = None,
    columns: list[str] = [],
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
        >>> df = pl.DataFrame(
                {
                    "user_id": [0, 0, 0, 1, 1, 1, 0, 2],
                    "article_id": [
                        9604210,
                        9634540,
                        9640420,
                        9647983,
                        9647984,
                        9647981,
                        None,
                        None,
                    ],
                    "first_page_time": [
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 18),
                        datetime.datetime(2023, 2, 25),
                        datetime.datetime(2023, 2, 22),
                        datetime.datetime(2023, 2, 21),
                        datetime.datetime(2023, 2, 23),
                        datetime.datetime(2023, 2, 19),
                        datetime.datetime(2023, 2, 26),
                    ],
                    "read_times": [
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
        >>> columns = ["first_page_time", "read_times"]
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, columns=columns).collect()
            shape: (8, 8)
            ┌─────────┬────────────┬─────────────────────┬────────────┬─────────┬─────────────────────────────┬──────────────────┬───────────────────────────────────┐
            │ user_id ┆ article_id ┆ first_page_time     ┆ read_times ┆ nothing ┆ article_id_fixed            ┆ read_times_fixed ┆ first_page_time_fixed             │
            │ ---     ┆ ---        ┆ ---                 ┆ ---        ┆ ---     ┆ ---                         ┆ ---              ┆ ---                               │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64        ┆ f32     ┆ list[i64]                   ┆ list[i64]        ┆ list[datetime[μs]]                │
            ╞═════════╪════════════╪═════════════════════╪════════════╪═════════╪═════════════════════════════╪══════════════════╪═══════════════════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0          ┆ null    ┆ [9604210, 9634540]          ┆ [0, 2]           ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2          ┆ null    ┆ [9604210, 9634540]          ┆ [0, 2]           ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6          ┆ null    ┆ [9604210, 9634540]          ┆ [0, 2]           ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8          ┆ null    ┆ [9604210, 9634540]          ┆ [0, 2]           ┆ [2023-02-18 00:00:00, 2023-02-18… │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1          ┆ null    ┆ [9647984, 9647983, 9647981] ┆ [1, 13, 1]       ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13         ┆ null    ┆ [9647984, 9647983, 9647981] ┆ [1, 13, 1]       ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1          ┆ null    ┆ [9647984, 9647983, 9647981] ┆ [1, 13, 1]       ┆ [2023-02-21 00:00:00, 2023-02-22… │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1          ┆ null    ┆ null                        ┆ null             ┆ null                              │
            └─────────┴────────────┴─────────────────────┴────────────┴─────────┴─────────────────────────────┴──────────────────┴───────────────────────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1, columns=columns).collect()
            shape: (8, 8)
            ┌─────────┬────────────┬─────────────────────┬────────────┬─────────┬──────────────────┬──────────────────┬───────────────────────┐
            │ user_id ┆ article_id ┆ first_page_time     ┆ read_times ┆ nothing ┆ article_id_fixed ┆ read_times_fixed ┆ first_page_time_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---        ┆ ---     ┆ ---              ┆ ---              ┆ ---                   │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64        ┆ f32     ┆ list[i64]        ┆ list[i64]        ┆ list[datetime[μs]]    │
            ╞═════════╪════════════╪═════════════════════╪════════════╪═════════╪══════════════════╪══════════════════╪═══════════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0          ┆ null    ┆ [9634540]        ┆ [2]              ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2          ┆ null    ┆ [9634540]        ┆ [2]              ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6          ┆ null    ┆ [9634540]        ┆ [2]              ┆ [2023-02-18 00:00:00] │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8          ┆ null    ┆ [9634540]        ┆ [2]              ┆ [2023-02-18 00:00:00] │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1          ┆ null    ┆ [9647981]        ┆ [1]              ┆ [2023-02-23 00:00:00] │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13         ┆ null    ┆ [9647981]        ┆ [1]              ┆ [2023-02-23 00:00:00] │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1          ┆ null    ┆ [9647981]        ┆ [1]              ┆ [2023-02-23 00:00:00] │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1          ┆ null    ┆ null             ┆ null             ┆ null                  │
            └─────────┴────────────┴─────────────────────┴────────────┴─────────┴──────────────────┴──────────────────┴───────────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1).collect()
            shape: (8, 6)
            ┌─────────┬────────────┬─────────────────────┬────────────┬─────────┬──────────────────┐
            │ user_id ┆ article_id ┆ first_page_time     ┆ read_times ┆ nothing ┆ article_id_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---        ┆ ---     ┆ ---              │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64        ┆ f32     ┆ list[i64]        │
            ╞═════════╪════════════╪═════════════════════╪════════════╪═════════╪══════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0          ┆ null    ┆ [9634540]        │
            │ 0       ┆ 9634540    ┆ 2023-02-18 00:00:00 ┆ 2          ┆ null    ┆ [9634540]        │
            │ 0       ┆ null       ┆ 2023-02-19 00:00:00 ┆ 6          ┆ null    ┆ [9634540]        │
            │ 0       ┆ 9640420    ┆ 2023-02-25 00:00:00 ┆ 8          ┆ null    ┆ [9634540]        │
            │ 1       ┆ 9647984    ┆ 2023-02-21 00:00:00 ┆ 1          ┆ null    ┆ [9647981]        │
            │ 1       ┆ 9647983    ┆ 2023-02-22 00:00:00 ┆ 13         ┆ null    ┆ [9647981]        │
            │ 1       ┆ 9647981    ┆ 2023-02-23 00:00:00 ┆ 1          ┆ null    ┆ [9647981]        │
            │ 2       ┆ null       ┆ 2023-02-26 00:00:00 ┆ 1          ┆ null    ┆ null             │
            └─────────┴────────────┴─────────────────────┴────────────┴─────────┴──────────────────┘
        >>> create_fixed_history_aggr_columns(df.lazy(), dt_cutoff, 1).head(1).collect()
            shape: (1, 6)
            ┌─────────┬────────────┬─────────────────────┬────────────┬─────────┬──────────────────┐
            │ user_id ┆ article_id ┆ first_page_time     ┆ read_times ┆ nothing ┆ article_id_fixed │
            │ ---     ┆ ---        ┆ ---                 ┆ ---        ┆ ---     ┆ ---              │
            │ i64     ┆ i64        ┆ datetime[μs]        ┆ i64        ┆ f32     ┆ list[i64]        │
            ╞═════════╪════════════╪═════════════════════╪════════════╪═════════╪══════════════════╡
            │ 0       ┆ 9604210    ┆ 2023-02-18 00:00:00 ┆ 0          ┆ null    ┆ [9634540]        │
            └─────────┴────────────┴─────────────────────┴────────────┴─────────┴──────────────────┘
    """
    _check_columns_in_df(df, [USER_ID, IMPRESSION_TIMESTAMP, ARTICLE_ID] + columns)
    aggr_columns = list(set([ARTICLE_ID] + columns))

    df = df.sort(USER_ID, IMPRESSION_TIMESTAMP)
    df_history = (
        df.select(pl.all())
        .filter(pl.col(ARTICLE_ID).is_not_null())
        .filter(pl.col(IMPRESSION_TIMESTAMP) < dt_cutoff)
        .group_by(USER_ID)
        .agg(
            pl.col(aggr_columns).suffix(FIXED_SUFFIX),
        )
    )
    if history_size is not None:
        for col in aggr_columns:
            df_history = df_history.with_columns(
                pl.col(col + FIXED_SUFFIX).list.tail(history_size)
            )

    return df.join(df_history, on="user_id", how="left")


def add_session_id_and_next_items(
    df: pl.DataFrame,
    session_length: datetime.timedelta = datetime.timedelta(minutes=30),
    shift_columns: list[str] = [],
    prefix: str = "next_",
    tqdm_kwargs: dict = {},
):
    """
    Adding session IDs and shifting specified columns to create 'next_' features.

    This function processes a DataFrame to assign unique session IDs based on a specified session length and creates new columns by shifting existing columns.
    These new columns are intended to represent the 'next_' features in a session-based context, e.g. 'next_read_time'.

    Args:
        df (pl.DataFrame): The DataFrame to process.
        session_length (datetime.timedelta, optional): The length of a session, used to determine when a new session ID should be assigned.
            Defaults to 30 minutes.
        shift_columns (list[str], optional): A list of column names whose values will be shifted to create the 'next_' features.
            Defaults to an empty list. If empty, you will only enrich with 'session_id'.
        tqdm_disable (bool, optional): If True, disables the tqdm progress bar. Useful in environments where tqdm's output is undesirable.
            Defaults to False. This may take some time, might be worth seeing the progress.
        prefix (str, optional): The prefix to add to the shifted column names. Defaults to 'next_'.

    Returns:
        pl.DataFrame: A modified DataFrame with added session IDs and 'next_clicked' features.

    Example:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame(
                {
                    "user_id": [1, 1, 2, 2],
                    "first_page_time": [
                        datetime.datetime(year=2023, month=1, day=1, minute=0),
                        datetime.datetime(year=2023, month=1, day=1, minute=20),
                        datetime.datetime(year=2023, month=1, day=1, minute=0),
                        datetime.datetime(year=2023, month=1, day=1, minute=35),
                    ],
                    "read_time": [9, 5, 1, 10],
                }
            )
        >>> add_session_id_and_next_items(df, datetime.timedelta(minutes=30), shift_columns=['read_time'])
            shape: (4, 5)
            ┌─────────┬─────────────────────┬───────────┬────────────┬────────────────┐
            │ user_id ┆ first_page_time     ┆ read_time ┆ session_id ┆ next_read_time │
            │ ---     ┆ ---                 ┆ ---       ┆ ---        ┆ ---            │
            │ i64     ┆ datetime[μs]        ┆ i64       ┆ u32        ┆ i64            │
            ╞═════════╪═════════════════════╪═══════════╪════════════╪════════════════╡
            │ 1       ┆ 2023-01-01 00:00:00 ┆ 9         ┆ 0          ┆ 5              │
            │ 1       ┆ 2023-01-01 00:20:00 ┆ 5         ┆ 0          ┆ null           │
            │ 2       ┆ 2023-01-01 00:00:00 ┆ 1         ┆ 2          ┆ null           │
            │ 2       ┆ 2023-01-01 00:35:00 ┆ 10        ┆ 3          ┆ null           │
            └─────────┴─────────────────────┴───────────┴────────────┴────────────────┘
        >>> add_session_id_and_next_items(df, datetime.timedelta(minutes=60), shift_columns=['read_time'])
            shape: (4, 5)
            ┌─────────┬─────────────────────┬───────────┬────────────┬────────────────┐
            │ user_id ┆ first_page_time     ┆ read_time ┆ session_id ┆ next_read_time │
            │ ---     ┆ ---                 ┆ ---       ┆ ---        ┆ ---            │
            │ i64     ┆ datetime[μs]        ┆ i64       ┆ u32        ┆ i64            │
            ╞═════════╪═════════════════════╪═══════════╪════════════╪════════════════╡
            │ 1       ┆ 2023-01-01 00:00:00 ┆ 9         ┆ 0          ┆ 5              │
            │ 1       ┆ 2023-01-01 00:20:00 ┆ 5         ┆ 0          ┆ null           │
            │ 2       ┆ 2023-01-01 00:00:00 ┆ 1         ┆ 2          ┆ 10             │
            │ 2       ┆ 2023-01-01 00:35:00 ┆ 10        ┆ 2          ┆ null           │
            └─────────┴─────────────────────┴───────────┴────────────┴────────────────┘
    """
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    # =>
    df = df.with_row_count(GROUPBY_ID)
    SESSION_ID = "session_id"
    # => INCREMENTAL SESSION-ID:
    s_id = 0
    # => COLUMNS:
    next_shift_columns = [prefix + feat for feat in shift_columns]
    select_columns = list(
        set([USER_ID, IMPRESSION_TIMESTAMP, GROUPBY_ID] + shift_columns)
    )
    # =>
    df_concat = []
    #
    _check_columns_in_df(df, select_columns)

    for df_user in tqdm(
        df.select(select_columns).partition_by(by=USER_ID),
        disable=tqdm_kwargs.get("disable", False),
        ncols=tqdm_kwargs.get("ncols", 80),
    ):
        df_session = (
            df_user.sort(IMPRESSION_TIMESTAMP)
            .groupby_dynamic(IMPRESSION_TIMESTAMP, every=session_length)
            .agg(
                GROUPBY_ID,
                pl.col(shift_columns).shift(-1).prefix(prefix),
            )
            .with_row_count(SESSION_ID, offset=s_id)
        )
        #
        s_id += df_user.shape[0]
        df_concat.append(df_session)

    df_concat = (
        pl.concat(df_concat)
        .lazy()
        .select(GROUPBY_ID, SESSION_ID, pl.col(next_shift_columns))
        .explode(GROUPBY_ID, pl.col(next_shift_columns))
        .collect()
    )
    return df.join(df_concat, on=GROUPBY_ID, how="left").drop(GROUPBY_ID)

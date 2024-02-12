from pathlib import Path
import polars as pl
import numpy as np
import datetime
import random

from ebrec.utils.utils_python import generate_unique_name

from ebrec.utils.constants import DEFAULT_USER_COL, DEFAULT_ARTICLE_ID_COL


# NOTE to self; when doing the test function, use the same 'df' for the dynamic / static histories


def _check_columns_in_df(df: pl.DataFrame, columns: list[str]) -> None:
    """
    Checks whether all specified columns are present in a Polars DataFrame.
    Raises a ValueError if any of the specified columns are not present in the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        columns (list[str]): The names of the columns to check for.

    Returns:
        None.

    Examples:
    >>> df = pl.DataFrame({"user_id": [1], "first_name": ["J"]})
    >>> check_columns_in_df(df, columns=["user_id", "not_in"])
        ValueError: Invalid input provided. The dataframe does not contain columns ['not_in'].
    """
    columns_not_in_df = [col for col in columns if col not in df.columns]
    if columns_not_in_df:
        raise ValueError(
            f"Invalid input provided. The DataFrame does not contain columns {columns_not_in_df}."
        )


def _validate_equal_list_column_lengths(df: pl.DataFrame, col1: str, col2: str) -> bool:
    """
    Checks if the items in two list columns of a DataFrame have equal lengths.

    Args:
        df (pl.DataFrame): The DataFrame containing the list columns.
        col1 (str): The name of the first list column.
        col2 (str): The name of the second list column.

    Returns:
        bool: True if the items in the two list columns have equal lengths, False otherwise.

    Raises:
        None.

    >>> df = pl.DataFrame({
            'col1': [[1, 2, 3], [4, 5], [6]],
            'col2': [[10, 20], [30, 40, 50], [60, 70, 80]],
        })
    >>> _validate_equal_list_column_lengths(df, 'col1', 'col2')
        ValueError: Mismatch in the lengths of the number of items (row-based) between the columns: 'col1' and 'col2'. Please ensure equal lengths.
    >>> df = df.with_columns(pl.Series('col1', [[1, 2], [3, 4, 5], [6, 7, 8]]))
    >>> _validate_equal_list_column_lengths(df, 'col1', 'col2')
    """
    if not df.select(pl.col(col1).list.len() == pl.col(col2).list.len())[col1].all():
        raise ValueError(
            f"Mismatch in the lengths of the number of items (row-based) between the columns: '{col1}' and '{col2}'. Please ensure equal lengths."
        )


def from_dict_to_polars(dictionary: dict) -> pl.DataFrame:
    """
    When dealing with dictionary with intergers as keys
    Example:
    >>> dictionary = {1: "a", 2: "b"}
    >>> from_dict_to_polars(dictionary)
        shape: (2, 2)
        ┌──────┬────────┐
        │ keys ┆ values │
        │ ---  ┆ ---    │
        │ i64  ┆ str    │
        ╞══════╪════════╡
        │ 1    ┆ a      │
        │ 2    ┆ b      │
        └──────┴────────┘
    >>> pl.from_dict(dictionary)
        raise ValueError("Series name must be a string.")
            ValueError: Series name must be a string.
    """
    return pl.DataFrame(
        {"keys": list(dictionary.keys()), "values": list(dictionary.values())}
    )


def shuffle_rows(df: pl.DataFrame, seed: int = None) -> pl.DataFrame:
    """
    Shuffle the rows of a DataFrame. This methods allows for LazyFrame,
    whereas, 'df.sample(fraction=1)' is not compatible.

    Examples:
    >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    >>> shuffle_rows(df.lazy(), seed=123).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 3   ┆ 3   ┆ 3   │
        │ 2   ┆ 2   ┆ 2   │
        └─────┴─────┴─────┘
    >>> shuffle_rows(df.lazy(), seed=None).collect().sort("a")
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 2   ┆ 2   ┆ 2   │
        │ 3   ┆ 3   ┆ 3   │
        └─────┴─────┴─────┘

    Test_:
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df, seed=None).iter_rows()])
        True

    Note:
        Be aware that 'pl.all().shuffle()' shuffles columns-wise, i.e., with if pl.all().shuffle(None)
        each column's element are shuffled independently from each other (example might change with no seed):
    >>> df_ = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}).select(pl.all().shuffle(None)).sort("a")
    >>> df_
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 3   ┆ 1   │
        │ 2   ┆ 2   ┆ 3   │
        │ 3   ┆ 1   ┆ 2   │
        └─────┴─────┴─────┘
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df_, seed=None).iter_rows()])
        False
    """
    seed = seed if seed is not None else random.randint(1, 1e6)
    return df.select(pl.all().shuffle(seed))


def keep_unique_values_in_list(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Removes duplicate article IDs from the specified list column of a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame containing the list column with article IDs.
        column (str): The name of the list column containing article IDs.

    Returns:
        pl.DataFrame: A new DataFrame with the same columns as the input DataFrame, but with duplicate
        article IDs removed from the specified list column.

    Example:
        >>> df = pl.DataFrame({
                "article_ids": [[1, 2, 3, 1, 2], [3, 4, 5, 3], [1, 2, 3, 1, 2, 3]],
                "hh": ["h", "e", "y"]
            })
        >>> keep_unique_values_in_list(df.lazy(), "article_ids").collect()
            shape: (3, 1)
            ┌─────────────┐
            │ article_ids │
            │ ---         │
            │ list[i64]   │
            ╞═════════════╡
            │ [1, 2, 3]   │
            │ [3, 4, 5]   │
            │ [1, 2, 3]   │
            └─────────────┘
    """
    return df.with_columns(pl.col(column).list.eval(pl.element().unique()))


def rank_list_ids_by_list_values(df, id_col: str, value_col: str) -> pl.DataFrame:
    """
    Ranks the list elements in the 'id_col' column of the DataFrame based on the corresponding values in the 'value_col' column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        id_col (str): The name of the column containing lists of IDs.
        value_col (str): The name of the column containing values corresponding to the IDs.


    Returns:
        pl.DataFrame: A DataFrame with the 'id_col' column sorted in ascending order based on the 'value_col' column.

    Raise:
        If id_col and value_col do not have the same number of values in each list.

    Returns:
        pl.DataFrame: _description_
    >>> df = pl.DataFrame(
            {
                "id": [["a", "b", "c"], ["a", "c", "b"]],
                "val": [[3, 2, 1], [10, 30, 20]],
                "h" : [1,2]
            }
        )
    >>> rank_list_ids_by_list_values(df, "id", "val")
        shape: (2, 3)
        ┌─────────────────┬──────────────┬─────┐
        │ id              ┆ val          ┆ h   │
        │ ---             ┆ ---          ┆ --- │
        │ list[str]       ┆ list[i64]    ┆ i64 │
        ╞═════════════════╪══════════════╪═════╡
        │ ["c", "b", "a"] ┆ [1, 2, 3]    ┆ 1   │
        │ ["a", "b", "c"] ┆ [10, 20, 30] ┆ 2   │
        └─────────────────┴──────────────┴─────┘
    """
    return df.with_columns(
        df.lazy()
        .select(pl.col(id_col), pl.col(value_col))
        .with_row_count("row_nr")
        .explode(pl.col(id_col, value_col))
        .sort(by=value_col)
        .group_by("row_nr")
        .agg(id_col, value_col)
        .sort("row_nr")
        .drop("row_nr")
        .collect()
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


def split_list_to_columns(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Converts a column of lists containing values to individual columns with the list values.

    Args:
        df (pl.DataFrame): The input DataFrame containing the list column.
        column (str): The name of the column to convert.

    Returns:
        pl.DataFrame: Transformed dataframe.

    Example:
    >>> df = pl.DataFrame({
            "list_col": [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [3, 5, 7],
            ],
            "other_col": [1, 2, 3, 4]
        })
    >>> split_list_to_columns(df, column="list_col")
        shape: (4, 4)
        ┌─────────┬─────────┬─────────┬───────────┐
        │ field_0 ┆ field_1 ┆ field_2 ┆ other_col │
        │ ---     ┆ ---     ┆ ---     ┆ ---       │
        │ i64     ┆ i64     ┆ i64     ┆ i64       │
        ╞═════════╪═════════╪═════════╪═══════════╡
        │ 1       ┆ 2       ┆ 3       ┆ 1         │
        │ 4       ┆ 5       ┆ 6       ┆ 2         │
        │ 7       ┆ 8       ┆ 9       ┆ 3         │
        │ 3       ┆ 5       ┆ 7       ┆ 4         │
        └─────────┴─────────┴─────────┴───────────┘
    """
    return df.with_columns(pl.col(column).list.to_struct()).unnest(column)


def compute_nested_lists_mean(
    df: pl.DataFrame, column: str, fill_null: list[float] = None
) -> pl.DataFrame:
    """
    Computes the mean of a column containing lists of float values and adds the result
    as a new column to the input DataFrame.

    Be aware, the method can handle different list lengths for 'column', however, zeros vectors should
    be removed prior.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column containing lists of float values.

    Returns:
        pl.DataFrame: DataFrame with the mean value of the input column

    Example:
    >>> df = pl.DataFrame(
            {
                "col1": [
                    [[1, 2], [1, 3], [1, 4]],
                    [[1, 3], [2, 4], [3, 5]],
                    [[2, 4], [4, 6]],
                    None,
                ],
                "col2": [4, 5, 6, 7],
            }
        )
    >>> df.dtypes
        [List(List(Int64)), Int64]
    >>> compute_nested_lists_mean(df, column="col1")
        shape: (4, 2)
        ┌──────────────┬──────┐
        │ col1         ┆ col2 │
        │ ---          ┆ ---  │
        │ list[f64]    ┆ i64  │
        ╞══════════════╪══════╡
        │ [1.0, 3.0]   ┆ 4    │
        │ [2.0, 4.0]   ┆ 5    │
        │ [3.0, 5.0]   ┆ 6    │
        │ [null, null] ┆ 7    │
        └──────────────┴──────┘
    >>> compute_nested_lists_mean(df, column="col1", fill_null=[0, 0])
        shape: (4, 2)
        ┌────────────┬──────┐
        │ col1       ┆ col2 │
        │ ---        ┆ ---  │
        │ list[f64]  ┆ i64  │
        ╞════════════╪══════╡
        │ [1.0, 3.0] ┆ 4    │
        │ [2.0, 4.0] ┆ 5    │
        │ [3.0, 5.0] ┆ 6    │
        │ [0.0, 0.0] ┆ 7    │
        └────────────┴──────┘
    """
    a = df.lazy().select(column).with_row_count("row_nr").explode(column).collect()
    # =>
    if fill_null is not None:
        a = a.with_columns(pl.col(column).fill_null(fill_null))
    # =>
    a = split_list_to_columns(a, column)
    # =>
    a = a.lazy().group_by("row_nr").mean().drop("row_nr")
    # =>
    a = a.select(pl.concat_list(a.columns).alias(column)).collect()
    return df.with_columns(a)


def compute_nested_lists_sum(
    df: pl.DataFrame, column: str, fill_null: list[float] = None
) -> pl.DataFrame:
    """
    Computes the sum of a column containing lists of float values and adds the result
    as a new column to the input DataFrame.

    Be aware, the method can handle different list lengths for 'column', however, zeros vectors should
    be removed prior.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column containing lists of float values.

    Returns:
        pl.DataFrame: DataFrame with the sum value of the input column

    Example:
    >>> df = pl.DataFrame(
            {
                "col1": [
                    [[1, 2], [1, 3], [1, 4]],
                    [[1, 3], [2, 4], [3, 5]],
                    [[2, 4], [4, 6]],
                    None,
                ],
                "col2": [4, 5, 6, 7],
            }
        )
    >>> df.dtypes
        [List(List(Int64)), Int64]
    >>> compute_nested_lists_sum(df, column="col1")
        shape: (4, 2)
        ┌───────────┬──────┐
        │ col1      ┆ col2 │
        │ ---       ┆ ---  │
        │ list[i64] ┆ i64  │
        ╞═══════════╪══════╡
        │ [3, 9]    ┆ 4    │
        │ [6, 12]   ┆ 5    │
        │ [6, 10]   ┆ 6    │
        │ [0, 0]    ┆ 7    │
        └───────────┴──────┘
    >>> compute_nested_lists_mean(df, column="col1", fill_null=[1, 1])
        shape: (4, 2)
        ┌────────────┬──────┐
        │ col1       ┆ col2 │
        │ ---        ┆ ---  │
        │ list[f64]  ┆ i64  │
        ╞════════════╪══════╡
        │ [1.0, 3.0] ┆ 4    │
        │ [2.0, 4.0] ┆ 5    │
        │ [3.0, 5.0] ┆ 6    │
        │ [1.0, 1.0] ┆ 7    │
        └────────────┴──────┘
    """
    a = df.select(column).with_row_count("row_nr").explode(column)
    # =>
    if fill_null is not None:
        a = a.with_columns(pl.col(column).fill_null(fill_null))
    # =>
    a = split_list_to_columns(a, column)
    # =>
    a = a.group_by("row_nr").sum().drop("row_nr")
    # =>
    a = a.select(pl.concat_list(a.columns).alias(column))
    return df.with_columns(a)


def filter_minimum_lengths_from_list(
    df: pl.DataFrame,
    n: int,
    column: str,
) -> pl.DataFrame:
    """Filters a DataFrame based on the minimum number of elements in an array column.

    Args:
        df (pl.DataFrame): The input DataFrame to filter.
        n (int): The minimum number of elements required in the array column.
        column (str): The name of the array column to filter on.

    Returns:
        pl.DataFrame: The filtered DataFrame.

    Example:
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "article_ids": [["a", "b", "c"], ["a", "b"], ["a"], ["a"]],
            }
        )
    >>> filter_minimum_lengths_from_list(df, n=2, column="article_ids")
        shape: (2, 2)
        ┌─────────┬─────────────────┐
        │ user_id ┆ article_ids     │
        │ ---     ┆ ---             │
        │ i64     ┆ list[str]       │
        ╞═════════╪═════════════════╡
        │ 1       ┆ ["a", "b", "c"] │
        │ 2       ┆ ["a", "b"]      │
        └─────────┴─────────────────┘
    >>> filter_minimum_lengths_from_list(df, n=None, column="article_ids")
        shape: (4, 2)
        ┌─────────┬─────────────────┐
        │ user_id ┆ article_ids     │
        │ ---     ┆ ---             │
        │ i64     ┆ list[str]       │
        ╞═════════╪═════════════════╡
        │ 1       ┆ ["a", "b", "c"] │
        │ 2       ┆ ["a", "b"]      │
        │ 3       ┆ ["a"]           │
        │ 4       ┆ ["a"]           │
        └─────────┴─────────────────┘
    """
    return (
        df.filter(df[column].list.len() >= n)
        if column in df and n is not None and n > 0
        else df
    )


def filter_maximum_lengths_from_list(
    df: pl.DataFrame,
    n: int,
    column: str,
) -> pl.DataFrame:
    """Filters a DataFrame based on the maximum number of elements in an array column.

    Args:
        df (pl.DataFrame): The input DataFrame to filter.
        n (int): The maximum number of elements required in the array column.
        column (str): The name of the array column to filter on.

    Returns:
        pl.DataFrame: The filtered DataFrame.

    Example:
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "article_ids": [["a", "b", "c"], ["a", "b"], ["a"], ["a"]],
            }
        )
    >>> filter_maximum_lengths_from_list(df, n=2, column="article_ids")
        shape: (3, 2)
        ┌─────────┬─────────────┐
        │ user_id ┆ article_ids │
        │ ---     ┆ ---         │
        │ i64     ┆ list[str]   │
        ╞═════════╪═════════════╡
        │ 2       ┆ ["a", "b"]  │
        │ 3       ┆ ["a"]       │
        │ 4       ┆ ["a"]       │
        └─────────┴─────────────┘
    >>> filter_maximum_lengths_from_list(df, n=None, column="article_ids")
        shape: (4, 2)
        ┌─────────┬─────────────────┐
        │ user_id ┆ article_ids     │
        │ ---     ┆ ---             │
        │ i64     ┆ list[str]       │
        ╞═════════╪═════════════════╡
        │ 1       ┆ ["a", "b", "c"] │
        │ 2       ┆ ["a", "b"]      │
        │ 3       ┆ ["a"]           │
        │ 4       ┆ ["a"]           │
        └─────────┴─────────────────┘
    """
    return (
        df.filter(df[column].list.len() <= n)
        if column in df and n is not None and n > 0
        else df
    )


def repeat_by_list_values(df: pl.DataFrame, repeat_column: str) -> pl.DataFrame:
    """
    This function can handle when the element to repeat is a list, as this is not supported at the moment.

    Args:
        df (pl.DataFrame): The input DataFrame.
        repeat_column (str): The name of the column containing the list values to repeat.

    Returns:
        pl.DataFrame: A new DataFrame with the list values repeated based on the corresponding 'repeat_column'.

    >>> df = (
            pl.DataFrame({
                'ID' : [100, 200],
                'val': [["a", "b", "c"], ["a", "b"]],
                'repeat_by': [2, 3],
            })
        )

    ## Polars does support the 'repeat_by' for intergers:
    >>> df.with_columns(pl.col("ID").repeat_by("repeat_by"))
        shape: (2, 3)
        ┌─────────────────┬─────────────────┬───────────┐
        │ ID              ┆ val             ┆ repeat_by │
        │ ---             ┆ ---             ┆ ---       │
        │ list[i64]       ┆ list[str]       ┆ i64       │
        ╞═════════════════╪═════════════════╪═══════════╡
        │ [100, 100]      ┆ ["a", "b", "c"] ┆ 2         │
        │ [200, 200, 200] ┆ ["a", "b"]      ┆ 3         │
        └─────────────────┴─────────────────┴───────────┘

    ## But not for list
    >>> df.with_columns(pl.col("val").repeat_by("repeat_by"))
        (...): `repeat_by` operation not supported for dtype `list[str]`

    ## This is where 'repeat_list_values' comes in:
    >>> repeat_list_values(df, "repeat_by")
        shape: (5, 3)
        ┌─────┬─────────────────┬───────────┐
        │ ID  ┆ val             ┆ repeat_by │
        │ --- ┆ ---             ┆ ---       │
        │ i64 ┆ list[str]       ┆ i64       │
        ╞═════╪═════════════════╪═══════════╡
        │ 100 ┆ ["a", "b", "c"] ┆ 2         │
        │ 100 ┆ ["a", "b", "c"] ┆ 2         │
        │ 200 ┆ ["a", "b"]      ┆ 3         │
        │ 200 ┆ ["a", "b"]      ┆ 3         │
        │ 200 ┆ ["a", "b"]      ┆ 3         │
        └─────┴─────────────────┴───────────┘
    """
    return df.with_columns(pl.col(repeat_column).repeat_by(repeat_column)).explode(
        repeat_column
    )


def drop_nulls_from_list(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Drops null values from a specified column in a Polars DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to drop null values from.

    Returns:
        pl.DataFrame: A new DataFrame with null values dropped from the specified column.

    Examples:
    >>> df = pl.DataFrame(
            {"user_id": [101, 102, 103], "dynamic_article_id": [[1, None, 3], None, [4, 5]]}
        )
    >>> print(df)
        shape: (3, 2)
        ┌─────────┬────────────────────┐
        │ user_id ┆ dynamic_article_id │
        │ ---     ┆ ---                │
        │ i64     ┆ list[i64]          │
        ╞═════════╪════════════════════╡
        │ 101     ┆ [1, null, 3]       │
        │ 102     ┆ null               │
        │ 103     ┆ [4, 5]             │
        └─────────┴────────────────────┘
    >>> drop_nulls_from_list(df, "dynamic_article_id")
        shape: (3, 2)
        ┌─────────┬────────────────────┐
        │ user_id ┆ dynamic_article_id │
        │ ---     ┆ ---                │
        │ i64     ┆ list[i64]          │
        ╞═════════╪════════════════════╡
        │ 101     ┆ [1, 3]             │
        │ 102     ┆ null               │
        │ 103     ┆ [4, 5]             │
        └─────────┴────────────────────┘
    """
    return df.with_columns(pl.col(column).list.eval(pl.element().drop_nulls()))


def concat_str_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "first_name": ["John", "Jane", "Alice"],
                "last_name": ["Doe", "Doe", "Smith"],
            }
        )
    >>> concatenated_df, concatenated_column_name = concat_str_columns(df, columns=['first_name', 'last_name'])
    >>> concatenated_df
        shape: (3, 4)
        ┌─────┬────────────┬───────────┬──────────────────────┐
        │ id  ┆ first_name ┆ last_name ┆ first_name-last_name │
        │ --- ┆ ---        ┆ ---       ┆ ---                  │
        │ i64 ┆ str        ┆ str       ┆ str                  │
        ╞═════╪════════════╪═══════════╪══════════════════════╡
        │ 1   ┆ John       ┆ Doe       ┆ John Doe             │
        │ 2   ┆ Jane       ┆ Doe       ┆ Jane Doe             │
        │ 3   ┆ Alice      ┆ Smith     ┆ Alice Smith          │
        └─────┴────────────┴───────────┴──────────────────────┘
    """
    concat_name = "-".join(columns)
    concat_columns = df.select(pl.concat_str(columns, separator=" ").alias(concat_name))
    return df.with_columns(concat_columns), concat_name


def filter_empty_text_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Example:
    >>> df = pl.DataFrame({"Name": ["John", "Alice", "Bob", ""], "Age": [25, 28, 30, 22]})
    >>> filter_empty_text_column(df, "Name")
        shape: (3, 2)
        ┌───────┬─────┐
        │ Name  ┆ Age │
        │ ---   ┆ --- │
        │ str   ┆ i64 │
        ╞═══════╪═════╡
        │ John  ┆ 25  │
        │ Alice ┆ 28  │
        │ Bob   ┆ 30  │
        └───────┴─────┘
    """
    return df.filter(df[column].str.lengths() > 0)


def cast_datetime(df: pl.DataFrame, column: str, **kwargs) -> pl.DataFrame:
    """
    >>> import datetime
    >>> from zoneinfo import ZoneInfo
    >>> df = pl.DataFrame(
            {
                "id": [1, 2],
                "time": [
                    datetime.datetime(2022, 10, 26, 13, 46, 48, tzinfo=ZoneInfo(key="UTC")),
                    datetime.datetime(2022, 10, 26, 13, 46, 57, tzinfo=ZoneInfo(key="UTC")),
                ],
            }
        )
    >>> cast_datetime(df, "time", **{"time_unit": "us"})
        shape: (2, 2)
        ┌─────┬─────────────────────┐
        │ id  ┆ time                │
        │ --- ┆ ---                 │
        │ i64 ┆ datetime[μs]        │
        ╞═════╪═════════════════════╡
        │ 1   ┆ 2022-10-26 13:46:48 │
        │ 2   ┆ 2022-10-26 13:46:57 │
        └─────┴─────────────────────┘
    """
    return (
        df.with_columns(pl.col(column).cast(pl.Utf8).str.to_datetime(**kwargs))
        if column in df
        else df
    )


def shuffle_list_column(
    df: pl.DataFrame, column: str, seed: int = None
) -> pl.DataFrame:
    """Shuffles the values in a list column of a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to shuffle.
        seed (int, optional): An optional seed value.
            Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column shuffled.

    Example:
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "list_col": [["a-", "b-", "c-"], ["a#", "b#", "c#"], ["a@", "b@", "c@"]],
                "rdn": ["h", "e", "y"],
            }
        )
    >>> shuffle_list_column(df, 'list_col', seed=1)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["c-", "b-", "a-"] ┆ h   │
        │ 2   ┆ ["a#", "b#", "c#"] ┆ e   │
        │ 3   ┆ ["c@", "b@", "a@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    No seed:
    >>> shuffle_list_column(df, 'list_col', seed=None)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["b-", "a-", "c-"] ┆ h   │
        │ 2   ┆ ["a#", "c#", "b#"] ┆ e   │
        │ 3   ┆ ["b@", "c@", "a@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )

    >>> df = pl.DataFrame({
            'id': [1, 2, 3],
            'list_col': [[6, 7, 8], [-6, -7, -8], [60, 70, 80]],
            'rdn': ['h', 'e', 'y']
        })
    >>> shuffle_list_column(df.lazy(), 'list_col', seed=2).collect()
        shape: (3, 3)
        ┌─────┬──────────────┬─────┐
        │ id  ┆ list_col     ┆ rdn │
        │ --- ┆ ---          ┆ --- │
        │ i64 ┆ list[i64]    ┆ str │
        ╞═════╪══════════════╪═════╡
        │ 1   ┆ [7, 6, 8]    ┆ h   │
        │ 2   ┆ [-8, -7, -6] ┆ e   │
        │ 3   ┆ [60, 80, 70] ┆ y   │
        └─────┴──────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )
    """
    _COLUMN_ORDER = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMN_ORDER, "_groupby_id")

    df = df.with_row_count(GROUPBY_ID)
    df_shuffle = (
        df.explode(column)
        .pipe(shuffle_rows, seed=seed)
        .group_by(GROUPBY_ID)
        .agg(column)
    )
    return (
        df.drop(column)
        .join(df_shuffle, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMN_ORDER)
    )


def split_df_in_n(df: pl.DataFrame, num_splits: int) -> list[pl.DataFrame]:
    """
    Split a DataFrame into n equal-sized splits.

    Args:
        df (pandas.DataFrame): The DataFrame to be split.
        num_splits (int): The number of splits to create.

    Returns:
        List[pandas.DataFrame]: A list of DataFrames, each representing a split.

    Examples:
        >>> df = pl.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7], "B" : [1, 2, 3, 4, 5, 6, 7]})
        >>> splits = split_df_in_n(df, 3)
        >>> for d in splits:
                print(d)
                shape: (3, 2)
                ┌─────┬─────┐
                │ A   ┆ B   │
                │ --- ┆ --- │
                │ i64 ┆ i64 │
                ╞═════╪═════╡
                │ 1   ┆ 1   │
                │ 2   ┆ 2   │
                │ 3   ┆ 3   │
                └─────┴─────┘
                shape: (3, 2)
                ┌─────┬─────┐
                │ A   ┆ B   │
                │ --- ┆ --- │
                │ i64 ┆ i64 │
                ╞═════╪═════╡
                │ 4   ┆ 4   │
                │ 5   ┆ 5   │
                │ 6   ┆ 6   │
                └─────┴─────┘
                shape: (1, 2)
                ┌─────┬─────┐
                │ A   ┆ B   │
                │ --- ┆ --- │
                │ i64 ┆ i64 │
                ╞═════╪═════╡
                │ 7   ┆ 7   │
                └─────┴─────┘

    """
    rows_per_split = int(np.ceil(df.shape[0] / num_splits))
    return [
        df[i * rows_per_split : (1 + i) * rows_per_split] for i in range(num_splits)
    ]


def concat_list_str(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Concatenate strings within lists for a specified column in a DataFrame.

    Args:
        df (polars.DataFrame): The input DataFrame.
        column (str): The name of the column in `df` that contains lists of strings
                        to be concatenated.

    Returns:
        polars.DataFrame: A DataFrame with the same structure as `df` but with the
                            specified column's lists of strings concatenated and
                            converted to a string instead of list.

    Examples:
        >>> df = pl.DataFrame({
                "strings": [["ab", "cd"], ["ef", "gh"], ["ij", "kl"]]
            })
        >>> concat_list_str(df, "strings")
            shape: (3, 1)
            ┌─────────┐
            │ strings │
            │ ---     │
            │ str     │
            ╞═════════╡
            │ ab cd   │
            │ ef gh   │
            │ ij kl   │
            └─────────┘
    """
    return df.with_columns(
        pl.col(column).list.eval(pl.element().str.concat(" "))
    ).explode(column)

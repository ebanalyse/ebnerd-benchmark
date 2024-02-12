try:
    import polars as pl
except ImportError:
    print("polars not available")


def linear_decay_weights(n: int, ascending: bool = True, **kwargs) -> list[float]:
    """
    Generates a list of weights in a linear decaying pattern.
    Args:
        n (int): The number of weights to generate. Must be a positive integer.
        ascending (bool, optional): Flag to determine the order of decay.
                                    If True, the decay is ascending. If False, it's descending.
                                    Defaults to True.
    Returns:
        List[float]: A list of linearly decaying weights.
    Raises:
        ValueError: If 'n' is not a positive integer.
    Examples:
    >>> linear_decay_weights(5, True)
        [0.2, 0.4, 0.6, 0.8, 1.0]
    >>> linear_decay_weights(10, False)
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    """
    weights = [(n - i) / n for i in range(n)]
    return weights if not ascending else weights[::-1]


def exponential_decay_weights(
    n: int, lambda_factor: float, ascending: bool = True, **kwargs
) -> list[float]:
    """
    Generates a list of weights in an exponential decay pattern.
    Args:
        n (int): The number of weights to generate. Must be a non-negative integer.
        lambda_factor (float): The factor by which the weights decay exponentially.
        ascending (bool, optional): Flag to determine the order of decay.
                                    If True, the decay is ascending. If False, it's descending.
                                    Defaults to True.
    Returns:
        List[float]: A list of exponentially decaying weights.
    Raises:
        ValueError: If 'n' is negative.
    Examples:
    >>> exponential_decay_weights(5, 0.5, True)
        [0.0625, 0.125, 0.25, 0.5, 1.0]
    >>> exponential_decay_weights(10, 0.5, False)
        [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125]
    """
    weights = [lambda_factor ** (n - i - 1) for i in range(n)]
    return weights if ascending else weights[::-1]


def add_decay_weights(
    df, column: str, decay_func: callable, ascending: bool = True, **kwargs: dict
):
    """
    Wrapper function: Adding decay weights to column using decay function scheme
    >>> df = pl.DataFrame(
            {
                "col1": [
                    [[1], [1], [1], [1]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1, 1], [1, 1, 1]],
                    None,
                ],
                "col2": [4, 5, 6, 7],
            }
        )
    >>> add_decay_weights(df, "col1", decay_func=linear_decay_weights, ascending=True)
        shape: (4, 3)
        ┌──────────────────────────┬───────────────────────────┬──────┐
        │ col1                     ┆ col1_weights              ┆ col2 │
        │ ---                      ┆ ---                       ┆ ---  │
        │ list[list[i64]]          ┆ list[f64]                 ┆ i64  │
        ╞══════════════════════════╪═══════════════════════════╪══════╡
        │ [[1], [1], … [1]]        ┆ [0.25, 0.5, … 1.0]        ┆ 4    │
        │ [[1, 1], [1, 1], [1, 1]] ┆ [0.333333, 0.666667, 1.0] ┆ 5    │
        │ [[1, 1, 1], [1, 1, 1]]   ┆ [0.5, 1.0]                ┆ 6    │
        │ null                     ┆ []                        ┆ 7    │
        └──────────────────────────┴───────────────────────────┴──────┘
    >>> add_decay_weights(df, "col1", decay_func=exponential_decay_weights, ascending=True, **{"lambda_factor" : 0.5})
        shape: (4, 3)
        ┌──────────────────────────┬──────────────────────┬──────┐
        │ col1                     ┆ col1_weights         ┆ col2 │
        │ ---                      ┆ ---                  ┆ ---  │
        │ list[list[i64]]          ┆ list[f64]            ┆ i64  │
        ╞══════════════════════════╪══════════════════════╪══════╡
        │ [[1], [1], … [1]]        ┆ [0.125, 0.25, … 1.0] ┆ 4    │
        │ [[1, 1], [1, 1], [1, 1]] ┆ [0.25, 0.5, 1.0]     ┆ 5    │
        │ [[1, 1, 1], [1, 1, 1]]   ┆ [0.5, 1.0]           ┆ 6    │
        │ null                     ┆ []                   ┆ 7    │
        └──────────────────────────┴──────────────────────┴──────┘
    """
    lengths = df[column].list.len().to_list()
    weights = [decay_func(n=i, ascending=ascending, **kwargs) for i in lengths]
    return df.with_columns(pl.Series(f"{column}_weights", weights))


def decay_weighting_nested_lists(
    df, column_history: str, column_history_weights: str, fill_nulls: int = None
):
    """
    >>> df = pl.DataFrame(
            {
                "col1": [
                    [[1], [1], [1], [1]],
                    [[1, 1], [1, 1], [1, 1]],
                    [[1, 1, 1], [1, 1, 1]],
                    [[1], None],
                    None,
                ],
                "col1_weights":
                    [[0.25, 0.5, 0.75, 1.0],
                    [0.33, 0.67, 1.0],
                    [0.5, 1.0],
                    [0.5, 1.0],
                    []
                ],
                "col2": [4, 5, 6, 7, 8 ],
            }
        )
    >>> decay_weighting_nested_lists(df, column_history="col1", column_history_weights="col1_weights")["col1"]
        Series: 'col1' [list[list[f64]]]
        [
            [[0.25], [0.5], … [1.0]]
            [[0.33, 0.33], [0.67, 0.67], [1.0, 1.0]]
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
            [[0.5], [null]]
            null
        ]
    >>> decay_weighting_nested_lists(df.lazy(), "col1", "col1_weights").collect()
    """
    GROUP_BY_COLUMN_FIRST = "group_by_1"
    GROUP_BY_COLUMN_SECOND = "group_by_2"
    COLUMNS = df.columns

    df = df.with_row_count(GROUP_BY_COLUMN_FIRST)

    exploded_weights = df.drop_nulls(column_history).select(
        pl.col(column_history_weights).explode()
    )

    if isinstance(exploded_weights, pl.LazyFrame):
        exploded_weights = exploded_weights.collect()

    df_ = (
        df.select(pl.col(GROUP_BY_COLUMN_FIRST, column_history))
        .drop_nulls(column_history)
        .explode(column_history)
        .with_columns(exploded_weights.select(column_history_weights))
        .with_row_count(GROUP_BY_COLUMN_SECOND)
        # Not optimal to explode, I want to compute [1,2,2] * 0.5 => (list * float)
        .explode(column_history)
        .with_columns(
            (pl.col(column_history) * pl.col(column_history_weights)).alias(
                column_history
            )
        )
        .group_by([GROUP_BY_COLUMN_SECOND])
        .agg(pl.col(GROUP_BY_COLUMN_FIRST).first(), column_history)
        .group_by(GROUP_BY_COLUMN_FIRST)
        .agg(column_history)
        .sort(GROUP_BY_COLUMN_FIRST)
    )

    return (
        df.drop(column_history)
        .join(df_, on=GROUP_BY_COLUMN_FIRST, how="left")
        .select(COLUMNS)
    )

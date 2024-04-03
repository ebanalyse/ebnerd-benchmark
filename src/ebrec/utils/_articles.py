from ebrec.utils._python import create_lookup_dict
import polars as pl
from ebrec.utils._constants import DEFAULT_ARTICLE_ID_COL

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers not available")


def load_article_id_embeddings(
    df: pl.DataFrame, path: str, item_col: str = DEFAULT_ARTICLE_ID_COL
) -> pl.DataFrame:
    """Load embeddings artifacts and join to articles on 'article_id'
    Args:
        path (str): Path to document embeddings
    """
    return df.join(pl.read_parquet(path), on=item_col, how="left")


def create_article_id_to_value_mapping(
    df: pl.DataFrame,
    value_col: str,
    article_col: str = DEFAULT_ARTICLE_ID_COL,
):
    return create_lookup_dict(
        df.select(article_col, value_col), key=article_col, value=value_col
    )


def convert_text2encoding_with_transformers(
    df: pl.DataFrame,
    tokenizer: AutoTokenizer,
    column: str,
    max_length: int = None,
) -> pl.DataFrame:
    """Converts text in a specified DataFrame column to tokens using a provided tokenizer.
    Args:
        df (pl.DataFrame): The input DataFrame containing the text column.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text. (from transformers import AutoTokenizer)
        column (str): The name of the column containing the text.
        max_length (int, optional): The maximum length of the encoded tokens. Defaults to None.
    Returns:
        pl.DataFrame: A new DataFrame with an additional column containing the encoded tokens.
    Example:
    >>> from transformers import AutoTokenizer
    >>> import polars as pl
    >>> df = pl.DataFrame({
            'text': ['This is a test.', 'Another test string.', 'Yet another one.']
        })
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> encoded_df, new_column = convert_text2encoding_with_transformers(df, tokenizer, 'text', max_length=20)
    >>> print(encoded_df)
        shape: (3, 2)
        ┌──────────────────────┬───────────────────────────────┐
        │ text                 ┆ text_encode_bert-base-uncased │
        │ ---                  ┆ ---                           │
        │ str                  ┆ list[i64]                     │
        ╞══════════════════════╪═══════════════════════════════╡
        │ This is a test.      ┆ [2023, 2003, … 0]             │
        │ Another test string. ┆ [2178, 3231, … 0]             │
        │ Yet another one.     ┆ [2664, 2178, … 0]             │
        └──────────────────────┴───────────────────────────────┘
    >>> print(new_column)
        text_encode_bert-base-uncased
    """
    text = df[column].to_list()
    # set columns
    new_column = f"{column}_encode_{tokenizer.name_or_path}"
    # If 'max_length' is provided then set it, else encode each string its original length
    padding = "max_length" if max_length else False
    encoded_tokens = tokenizer(
        text,
        add_special_tokens=False,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )["input_ids"]
    return df.with_columns(pl.Series(new_column, encoded_tokens)), new_column


def create_sort_based_prediction_score(
    df: pl.DataFrame,
    column: str,
    desc: bool,
    article_id_col: str = DEFAULT_ARTICLE_ID_COL,
    prediction_score_col: str = "prediction_score",
) -> pl.DataFrame:
    """
    Generates a prediction score for each row in a Polars DataFrame based on the sorting of a specified column.

    Args:
        df (pl.DataFrame): The input DataFrame to process.
        column (str): The name of the column to sort by and to base the prediction scores on.
        desc (bool): Determines the sorting order. If True, sort in descending order; otherwise, in ascending order.
        article_id_col (str, optional): The name article ID column. Defaults to "article_id".
        prediction_score_col (str, optional): The name to assign to the prediction score column. Defaults to "prediction_score".

    Returns:
        pl.DataFrame: A Polars DataFrame including the original data along with the new prediction score column.

    Examples:
    >>> import polars as pl
    >>> df = pl.DataFrame({
            "article_id": [1, 2, 3, 4, 5],
            "views": [100, 150, 200, 50, 300],
        })
    >>> create_sort_based_prediction_score(df, "views", True)
        shape: (5, 3)
        ┌────────────┬───────┬──────────────────┐
        │ article_id ┆ views ┆ prediction_score │
        │ ---        ┆ ---   ┆ ---              │
        │ i64        ┆ i64   ┆ f64              │
        ╞════════════╪═══════╪══════════════════╡
        │ 5          ┆ 300   ┆ 1.0              │
        │ 3          ┆ 200   ┆ 0.5              │
        │ 2          ┆ 150   ┆ 0.333333         │
        │ 1          ┆ 100   ┆ 0.25             │
        │ 4          ┆ 50    ┆ 0.2              │
        └────────────┴───────┴──────────────────┘
    """
    _TEMP_NAME = "index"
    return (
        (
            df.select(article_id_col, column)
            .sort(by=column, descending=desc)
            .with_row_index(name=_TEMP_NAME, offset=1)
        )
        .with_columns((1 / pl.col(_TEMP_NAME)).alias(prediction_score_col))
        .drop(_TEMP_NAME)
    )

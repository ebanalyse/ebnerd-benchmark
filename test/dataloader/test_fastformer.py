from pathlib import Path
import polars as pl
import numpy as np
import torch
from ebrec.utils._behaviors import create_user_id_to_int_mapping
from ebrec.utils._articles import create_article_id_to_value_mapping

from ebrec.utils._python import time_it
from ebrec.utils._behaviors import create_binary_labels_column
from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_USER_COL,
)

from ebrec.models.fastformer.dataloader import FastformerDataset
from torch.utils.data import DataLoader

TOKEN_COL = "tokens"
N_SAMPLES = "n"
BATCH_SIZE = 100

# LOAD DATA:
PATH_DATA = Path("test/data")
df_articles = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "articles.parquet"))
    .select(pl.col(DEFAULT_ARTICLE_ID_COL, DEFAULT_CATEGORY_COL))
    .with_columns(pl.Series(TOKEN_COL, np.random.randint(0, 20, (1, 10))))
    .collect()
)
df_history = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "history.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
    .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))
)
df_behaviors = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "behaviors.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL)
    .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias(N_SAMPLES))
    .join(df_history, on=DEFAULT_USER_COL, how="left")
    .collect()
    .pipe(create_binary_labels_column)
)
# => MAPPINGS:
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=TOKEN_COL
)
user_mapping = create_user_id_to_int_mapping(df=df_behaviors)
# => NPRATIO IMPRESSION - SAME LENGTHS:
df_behaviors_train = df_behaviors.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())
# => FOR TEST-DATALOADER
label_lengths = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()


@time_it(True)
def test_FastformerDataloader():
    train_dataloader = DataLoader(
        FastformerDataset(
            behaviors=df_behaviors_train,
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            article_dict=article_mapping,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    batch = train_dataloader.__iter__().__next__()

    assert train_dataloader.__len__() == int(np.ceil(df_behaviors_train.shape[0] / 100))
    assert len(batch) == 2, "There should be two outputs: (inputs, labels)"
    assert (
        len(batch[0]) == 2
    ), "Fastformer has two outputs (history_input, candidate_input)"

    for type_in_batch in batch[0]:
        assert (
            type_in_batch.dtype == torch.int
        ), "Expected output to be integer; used for lookup value"

    assert batch[1].dtype == torch.float, "Expected output to be integer; this is label"

    test_dataloader = DataLoader(
        FastformerDataset(
            behaviors=df_behaviors,
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            article_dict=article_mapping,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    batch = test_dataloader.__iter__().__next__()
    assert len(batch[1].squeeze(0)) == sum(
        label_lengths[:BATCH_SIZE]
    ), "Should have unfolded all the test samples"

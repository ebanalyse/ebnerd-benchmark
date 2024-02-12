from pathlib import Path
import polars as pl
import numpy as np
import torch
from ebrec.utils.utils_behaviors import create_user_id_mapping
from ebrec.utils.utils_articles import create_title_mapping
from ebrec.utils.utils_python import create_lookup_dict

from ebrec.models.newsrec.dataloader import (
    LSTURDataLoader,
    NAMLDataLoader,
    NRMSDataLoader,
)
from ebrec.utils.utils_python import time_it
from torch.utils.data import DataLoader

from ebrec.utils.utils_behaviors import create_binary_labels_column
from ebrec.utils.constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_USER_COL,
)

from ebrec.models.fastformer.dataloader import FastformerDataset
from torch.utils.data import DataLoader

N_SAMPLES = "n"
TOKEN_COL = "idx"
BATCH_SIZE = 100

# LOAD DATA:
PATH_DATA = Path("test/data")

df_articles = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "articles.parquet"))
    .select(pl.col(DEFAULT_ARTICLE_ID_COL, DEFAULT_CATEGORY_COL))
    .collect()
)
dummy_lookups = np.random.randint(0, 20, (df_articles.shape[0], 10))
df_articles = df_articles.with_columns(pl.Series(TOKEN_COL, dummy_lookups))
# =>
df_history = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "history.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
    .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))
    .collect()
)
df_behaviors = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "behaviors.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL)
    .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias(N_SAMPLES))
    .collect()
    .pipe(create_binary_labels_column)
    .join(df_history, on=DEFAULT_USER_COL, how="left")
)

#
user_mapping = create_user_id_mapping(df=df_behaviors)
article_mapping = create_title_mapping(df=df_articles, column=TOKEN_COL)
# =>
df_behaviors_train = df_behaviors.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())
# => ASSERT
label_lengths = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()


# ===
@time_it(True)
def test_NRMSDataLoader():
    train_dataloader = NRMSDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=False,
        batch_size=BATCH_SIZE,
    )

    batch = train_dataloader.__iter__().__next__()

    assert train_dataloader.__len__() == int(np.ceil(df_behaviors_train.shape[0] / 100))
    assert len(batch) == 2, "There should be two outputs: (inputs, labels)"
    assert (
        len(batch[0]) == 2
    ), "NRMS has two outputs (his_input_title, pred_input_title_one)"

    for type_in_batch in batch[0][0]:
        assert isinstance(
            type_in_batch.ravel()[0], np.integer
        ), "Expected output to be integer; used for lookup value"

    assert isinstance(
        batch[1].ravel()[0], np.integer
    ), "Expected output to be integer; this is label"

    test_dataloader = NRMSDataLoader(
        behaviors=df_behaviors,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=True,
        batch_size=BATCH_SIZE,
    )

    batch = test_dataloader.__iter__().__next__()
    assert len(batch[1]) == sum(
        label_lengths[:BATCH_SIZE]
    ), "Should have unfolded all the test samples"


@time_it(True)
def test_LSTURDataLoader():
    train_dataloader = LSTURDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        user_id_mapping=user_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        batch_size=BATCH_SIZE,
    )

    batch = train_dataloader.__iter__().__next__()

    assert train_dataloader.__len__() == int(np.ceil(df_behaviors_train.shape[0] / 100))
    assert len(batch) == 2, "There should be two outputs: (inputs, labels)"
    assert (
        len(batch[0]) == 3
    ), "LSTUR has two outputs (user_indexes, his_input_title, pred_input_title_one)"

    for type_in_batch in batch[0][0]:
        assert isinstance(
            type_in_batch.ravel()[0], np.integer
        ), "Expected output to be integer; used for lookup value"

    assert isinstance(
        batch[1].ravel()[0], np.integer
    ), "Expected output to be integer; this is label"

    test_dataloader = LSTURDataLoader(
        behaviors=df_behaviors,
        article_dict=article_mapping,
        user_id_mapping=user_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        batch_size=BATCH_SIZE,
        eval_mode=True,
    )

    batch = test_dataloader.__iter__().__next__()
    assert len(batch[1]) == sum(
        label_lengths[:BATCH_SIZE]
    ), "Should have unfolded all the test samples"


@time_it(True)
def test_NAMLDataLoader():
    body_mapping = article_mapping
    category_mapping = create_lookup_dict(
        df_articles.select(pl.col(DEFAULT_CATEGORY_COL).unique()).with_row_index(
            "row_nr"
        ),
        key=DEFAULT_CATEGORY_COL,
        value="row_nr",
    )
    subcategory_mapping = category_mapping

    train_dataloader = NAMLDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        body_mapping=body_mapping,
        category_mapping=category_mapping,
        unknown_representation="zeros",
        subcategory_mapping=subcategory_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        batch_size=BATCH_SIZE,
    )

    batch = train_dataloader.__iter__().__next__()

    assert train_dataloader.__len__() == int(np.ceil(df_behaviors_train.shape[0] / 100))
    assert len(batch) == 2, "There should be two outputs: (inputs, labels)"
    assert (
        len(batch[0]) == 8
    ), "NAML has two outputs (his_input_title,his_input_body,his_input_vert,his_input_subvert,pred_input_title,pred_input_body,pred_input_vert,pred_input_subvert)"

    for type_in_batch in batch[0][0]:
        assert isinstance(
            type_in_batch.ravel()[0], np.integer
        ), "Expected output to be integer; used for lookup value"

    assert isinstance(
        batch[1].ravel()[0], np.integer
    ), "Expected output to be integer; this is label"


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

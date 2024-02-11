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

# LOAD DATA:
PATH_DATA = Path("test/data")
df_embeddings = pl.read_parquet(PATH_DATA.joinpath("ebnerd", "document_vector.parquet"))
df_articles = pl.read_parquet(PATH_DATA.joinpath("ebnerd", "articles.parquet"))
df_behaviors = pl.read_parquet(PATH_DATA.joinpath("ebnerd", "behaviors.parquet"))
df_history = pl.read_parquet(PATH_DATA.joinpath("ebnerd", "history.parquet"))


label_lengths = df_behaviors["labels"].list.len().to_list()
article_mapping = create_title_mapping(df=df_articles, column="as")

BATCH_SIZE = 100
HISTORY_COLUMN = "article_id_fixed"


# ===
@time_it(True)
def test_NRMSDataLoader():
    article_mapping = create_title_mapping(df=df_articles, column=TOKENIZER_NAME)
    train_dataloader = NRMSDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        history_column=HISTORY_COLUMN,
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
        behaviors=df_behaviors_test,
        article_dict=article_mapping,
        history_column=HISTORY_COLUMN,
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
    article_mapping = create_title_mapping(df=df_articles, column=TOKENIZER_NAME)
    user_mapping = create_user_id_mapping(df=df_behaviors_train)

    train_dataloader = LSTURDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        user_id_mapping=user_mapping,
        history_column=HISTORY_COLUMN,
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
        behaviors=df_behaviors_test,
        article_dict=article_mapping,
        user_id_mapping=user_mapping,
        history_column=HISTORY_COLUMN,
        unknown_representation="zeros",
        batch_size=BATCH_SIZE,
        eval_mode=True,
    )

    batch = test_dataloader.__iter__().__next__()
    assert len(batch[1]) == sum(
        label_lengths[:BATCH_SIZE]
    ), "Should have unfolded all the test samples"


@time_it(True)
def test_FastformerDataloader():
    article_mapping = create_title_mapping(df=df_articles, column=TOKENIZER_NAME)

    train_dataloader = DataLoader(
        FastformerDataset(
            behaviors=df_behaviors_train,
            history_column=HISTORY_COLUMN,
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
            behaviors=df_behaviors_test,
            history_column=HISTORY_COLUMN,
            article_dict=article_mapping,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    batch = test_dataloader.__iter__().__next__()
    assert len(batch[1].squeeze(0)) == sum(
        label_lengths[:BATCH_SIZE]
    ), "Should have unfolded all the test samples"


@time_it(True)
def test_NAMLDataLoader():
    article_mapping = create_title_mapping(df=df_articles, column=TOKENIZER_NAME)
    body_mapping = article_mapping
    category_mapping = create_lookup_dict(
        df_articles.select(pl.col("category").unique()).with_row_count(),
        key="category",
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
        history_column=HISTORY_COLUMN,
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

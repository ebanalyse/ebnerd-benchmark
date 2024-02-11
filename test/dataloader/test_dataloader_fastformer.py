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

from ebrec.models.fastformer.dataloader import FastformerDataset
from torch.utils.data import DataLoader


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

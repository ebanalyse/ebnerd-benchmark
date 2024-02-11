from pathlib import Path
import polars as pl
import numpy as np

from ebrec.downloads.models.newsrec.dataloader import (
    NRMSDataLoader,
    LSTURDataLoader,
    NAMLDataLoader,
)

# from ebrec.newsrec.dataloader import NRMSDataLoader, LSTURDataLoader, NAMLDataLoader
# from models.newsrec.utils import create_title_mapping, create_user_id_mapping
# from newssources.utils_polars import create_lookup_dict
# from pytest.utils import timer_decorator
from tqdm import tqdm

# from models.fastformer.dataloader import FastformerDataset
# from torch.utils.data import DataLoader

EMBEDDING_NAME = f"title-subtitle-bert-base-multilingual-cased"
TOKENIZER_NAME = f"title_encode_bert-base-multilingual-cased"
PATH_DATA = Path("pytest/data")

df_articles = pl.read_parquet(PATH_DATA.joinpath("articles_data_with_emb.parquet"))
df_behaviors_train = pl.read_parquet(
    PATH_DATA.joinpath("behaviors_formatted_train.parquet")
)
df_behaviors_test = pl.read_parquet(
    PATH_DATA.joinpath("behaviors_formatted_test.parquet")
)

BATCH_SIZE = 100
HISTORY_COLUMN = "article_id_fixed"

label_lengths = df_behaviors_test["labels"].list.len().to_list()

N_ITERATIONS = 300

article_mapping = create_title_mapping(df=df_articles, column=TOKENIZER_NAME)


def iter_dataloader(dataloader, name: str, iterations: int):
    for _ in tqdm(range(iterations), desc=name):
        for _ in dataloader:
            pass


# ===
@timer_decorator
def bomb_NRMSDataLoader():
    dataloader = NRMSDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        history_column=HISTORY_COLUMN,
        unknown_representation="zeros",
        eval_mode=False,
        batch_size=BATCH_SIZE,
    )
    iter_dataloader(dataloader, "NRMS-train", iterations=N_ITERATIONS)

    dataloader = NRMSDataLoader(
        behaviors=df_behaviors_test,
        article_dict=article_mapping,
        history_column=HISTORY_COLUMN,
        unknown_representation="zeros",
        eval_mode=True,
        batch_size=BATCH_SIZE,
    )
    iter_dataloader(dataloader, "NRMS-test", iterations=N_ITERATIONS)


@timer_decorator
def bomb_LSTURDataLoader():
    user_mapping = create_user_id_mapping(df=df_behaviors_train)

    dataloader = LSTURDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        user_id_mapping=user_mapping,
        history_column=HISTORY_COLUMN,
        unknown_representation="zeros",
        batch_size=BATCH_SIZE,
    )
    iter_dataloader(dataloader, "LSTUR-train", iterations=N_ITERATIONS)

    dataloader = LSTURDataLoader(
        behaviors=df_behaviors_test,
        article_dict=article_mapping,
        user_id_mapping=user_mapping,
        history_column=HISTORY_COLUMN,
        unknown_representation="zeros",
        batch_size=BATCH_SIZE,
        eval_mode=True,
    )
    iter_dataloader(dataloader, "LSTUR-test", iterations=N_ITERATIONS)


# ===
@timer_decorator
def bomb_FastformerDataLoader():
    dataloader = DataLoader(
        FastformerDataset(
            behaviors=df_behaviors_train,
            history_column=HISTORY_COLUMN,
            article_dict=article_mapping,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )
    iter_dataloader(dataloader, "Fastformer-train", iterations=N_ITERATIONS)

    dataloader = DataLoader(
        FastformerDataset(
            behaviors=df_behaviors_test,
            history_column=HISTORY_COLUMN,
            article_dict=article_mapping,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )
    iter_dataloader(dataloader, "Fastformer-test", iterations=N_ITERATIONS)

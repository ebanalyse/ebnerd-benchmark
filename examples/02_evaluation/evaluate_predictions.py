from src.ebrec.evaluation.beyond_accuracy import (
    IntralistDiversity,
    Distribution,
    Serendipity,
    Novelty,
)
from src.ebrec.utils._python import parse_line

from pathlib import Path
import polars as pl
import numpy as np


def get_top_n_values(l1: np.ndarray, l2: np.ndarray, n: int) -> np.ndarray:
    """
    >>> get_top_n_values(l1=[11, 12, 13, 14, 15, 16], l2=[1, 3, 5, 6, 4, 2], n=3)
        array([11, 16, 12])
    """
    return np.asarray(l1)[np.argsort(l2)[:n]]


def create_lookup_dict(df: pl.DataFrame, key: str, value: str):
    return dict(zip(df[key], df[value]))


PATH = Path("/Users/johannes.kruse/Desktop/datasets/w23/small")

df = pl.read_parquet(PATH.joinpath("test", "behaviors.parquet"))
df_ba = df.filter("is_beyond_accuracy")[:1000]
candidate_list = df_ba[0]["article_ids_inview"][0].to_list()
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))

df_emb = pl.read_parquet(
    PATH.parent.joinpath("enrichments/embeddings/document_vector.parquet")
)
df_articles = df_articles.join(df_emb, on="article_id", how="inner")

df_articles.columns
d = {}
cols = ["document_vector", "topics", "category"]
for row in df_articles.iter_slices(n_rows=1):
    vals = {c: row[c].to_list()[0] for c in cols}
    d.update({row["article_id"][0]: vals})


top_n = 5
top_picks = []
a = []
with open(PATH.joinpath("test", "predictions_random.txt"), "r") as file:
    for l in file:
        idx, ranks = parse_line(l)
        if idx != "0":
            continue
        top_picks.append(get_top_n_values(candidate_list, ranks, top_n))

# RANDOM
div = IntralistDiversity()
res = div(R=top_picks, lookup_dict=d, lookup_key="document_vector")
res.mean()
#
df_articles.columns

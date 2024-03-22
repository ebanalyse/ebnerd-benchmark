from ebrec.evaluation.beyond_accuracy import (
    IntralistDiversity,
    Distribution,
    Serendipity,
    Novelty,
)
from ebrec.utils._constants import (
    DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
    DEFAULT_ARTICLE_MODIFIED_TIMESTAMP_COL,
    DEFAULT_TOTAL_PAGEVIEWS_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_TOTAL_INVIEWS_COL,
    DEFAULT_ARTICLE_ID_COL,
)
from pathlib import Path
import polars as pl
import numpy as np
import json

path = Path("examples/downloads/large")
path_beyond = path.joinpath("beyond_accuracy")
path_beyond.mkdir(exist_ok=True, parents=True)

CANDIDATE_LIST = "candidate_list.json"
CANDIDATE_DICT = "candidate_dict.json"

#
df_behaviors = (
    pl.scan_parquet(path.joinpath("test", "behaviors.parquet"))
    .filter(pl.col("is_beyond_accuracy"))
    .collect()
)
df_articles = pl.scan_parquet(path.joinpath("articles.parquet"))
# ======================================================================
### Make candidate list for beyond-accuracy:
candidate_list = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL][0].to_list()

with open(path_beyond.joinpath(CANDIDATE_LIST), "w") as f:
    json.dump(candidate_list, f)  # Dumping the list as JSON

with open(path_beyond.joinpath(CANDIDATE_LIST), "r") as f:
    load_candidate_list = json.load(f)

# Just checking in:
if (
    not (df_behaviors[DEFAULT_INVIEW_ARTICLES_COL] == candidate_list).sum()
    == df_behaviors.shape[0]
):
    raise ValueError("candidate_list is not identical in the testset")
if not (np.array(candidate_list) - np.array(load_candidate_list)).sum() == 0:
    raise ValueError("candidate_list was not dump correctly")

# ======================================================================
### Make candidate lookup dictionary for beyond-accuracy:
# => Embeddings:
emb_contrastive = pl.scan_parquet(
    path.parent.joinpath(
        "embeddings/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet"
    )
)
emb_roberta = pl.scan_parquet(
    path.parent.joinpath(
        "embeddings/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet"
    )
)
emb_bert = pl.scan_parquet(
    path.parent.joinpath(
        "embeddings/google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet"
    )
)
emb_docvec = pl.scan_parquet(
    path.parent.joinpath("embeddings/Ekstra_Bladet_word2vec/document_vector.parquet")
)
# =>
candidate_articles = (
    df_articles.filter(pl.col(DEFAULT_ARTICLE_ID_COL).is_in(candidate_list))
    .join(emb_contrastive, on=DEFAULT_ARTICLE_ID_COL, how="inner")
    .join(emb_roberta, on=DEFAULT_ARTICLE_ID_COL, how="inner")
    .join(emb_bert, on=DEFAULT_ARTICLE_ID_COL, how="inner")
    .join(emb_docvec, on=DEFAULT_ARTICLE_ID_COL, how="inner")
    .with_columns(
        pl.col(
            DEFAULT_ARTICLE_MODIFIED_TIMESTAMP_COL,
            DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
        ).cast(pl.Utf8)
    )
    # Zeros might cause issues
    .with_columns(
        pl.col(DEFAULT_TOTAL_INVIEWS_COL, DEFAULT_TOTAL_PAGEVIEWS_COL).fill_null(1)
    )
    .collect()
)
candidate_articles.columns
# Make lookup-dictionary:
candidate_dict = {}
for row in candidate_articles.iter_rows(named=True):
    # Note, all keys in dictionaries are converted to strings, when serializing an object to JSON format.
    candidate_dict[str(row[DEFAULT_ARTICLE_ID_COL])] = row
# Write it:
with open(path_beyond.joinpath(CANDIDATE_DICT), "w") as f:
    json.dump(candidate_dict, f)  # Dumping the list as JSON

with open(path_beyond.joinpath(CANDIDATE_DICT), "r") as f:
    load_candidate_dict = json.load(f)

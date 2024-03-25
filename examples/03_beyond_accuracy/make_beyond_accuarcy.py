from ebrec.utils._python import write_json_file, read_json_file
from pathlib import Path
import polars as pl
import numpy as np
import json

from ebrec.utils._constants import (
    DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
    DEFAULT_ARTICLE_MODIFIED_TIMESTAMP_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_TOTAL_PAGEVIEWS_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_TOTAL_INVIEWS_COL,
    DEFAULT_IS_SUBSCRIBER_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_POSTCODE_COL,
    DEFAULT_GENDER_COL,
    DEFAULT_AGE_COL,
)

path = Path("examples/downloads/demo")
path_beyond = path.joinpath("beyond_accuracy")
path_beyond.mkdir(exist_ok=True, parents=True)

CANDIDATE_LIST = "candidate_list.json"
CANDIDATE_DICT = "candidate_dict.json"
USERS_DICT = "users_dict.json"
BEHAVIORS_TIMESTAMP_DICT = "behaviors_timestamp_dict.json"

#
df_behaviors = (
    pl.scan_parquet(path.joinpath("test", "behaviors.parquet"))
    .filter(pl.col("is_beyond_accuracy"))
    .collect()
)
df_articles = pl.scan_parquet(path.joinpath("articles.parquet"))

# DUMP META DATA FOR USERS:
users_dict = {
    DEFAULT_IS_SUBSCRIBER_COL: df_behaviors[DEFAULT_IS_SUBSCRIBER_COL].to_list(),
    DEFAULT_POSTCODE_COL: df_behaviors[DEFAULT_POSTCODE_COL].to_list(),
    DEFAULT_GENDER_COL: df_behaviors[DEFAULT_GENDER_COL].to_list(),
    DEFAULT_AGE_COL: df_behaviors[DEFAULT_AGE_COL].to_list(),
}
write_json_file(users_dict, path_beyond.joinpath(USERS_DICT))

# Behaviors-timestamp:
df_behaviors_timestamp = (
    pl.scan_parquet(path.joinpath("test", "behaviors.parquet"))
    .filter(~pl.col("is_beyond_accuracy"))
    .select(
        pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).cast(pl.Utf8),
    )
    .collect()
)
behaviors_timestamp_dict = {
    DEFAULT_IMPRESSION_TIMESTAMP_COL: df_behaviors_timestamp[
        DEFAULT_IMPRESSION_TIMESTAMP_COL
    ].to_list()
}
write_json_file(
    behaviors_timestamp_dict, path_beyond.joinpath(BEHAVIORS_TIMESTAMP_DICT)
)

# ======================================================================
### Make candidate list for beyond-accuracy:
candidate_list = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL][0].to_list()

write_json_file(candidate_list, path_beyond.joinpath(CANDIDATE_LIST))
load_candidate_list = read_json_file(path_beyond.joinpath(CANDIDATE_LIST))

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
write_json_file(candidate_dict, path_beyond.joinpath(CANDIDATE_DICT))

# MAKE BASELINES
from ebrec.utils._python import write_submission_file, rank_predictions_by_score


# Editorial pick:
column = DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL
desc = False
# Editorial pick:
column = DEFAULT_TOTAL_INVIEWS_COL
desc = True
# Popular:
column = DEFAULT_TOTAL_PAGEVIEWS_COL
desc = True


PRED_NAME = "prediction_pick"
candidate_articles_ = (
    candidate_articles.select(DEFAULT_ARTICLE_ID_COL, column)
    .sort(by=column, descending=desc)
    .with_row_index(name="prediction_pick", offset=1)
)

mapping = {
    aid: value
    for aid, value in candidate_articles_.select(
        DEFAULT_ARTICLE_ID_COL, PRED_NAME
    ).iter_rows()
}

candidate_list_views = rank_predictions_by_score(
    [mapping.get(aid, 0) for aid in candidate_list]
)
candidate_list_views = [mapping.get(aid, 0) for aid in candidate_list]

#
df_behaviors = df_behaviors.with_columns(
    pl.Series(DEFAULT_INVIEW_ARTICLES_COL, [candidate_list_views])
)
#
ids = df_behaviors["impression_id"].to_list()
preds = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL].to_list()

write_submission_file(
    impression_ids=ids,
    prediction_scores=preds,
    path=path_beyond.joinpath("predictions.txt"),
    filename_zip="predictions_inviews.zip",
)

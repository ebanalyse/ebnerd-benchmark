from ebrec.utils._python import write_json_file, read_json_file, create_lookup_dict
from pathlib import Path
import polars as pl
import numpy as np
import json

from ebrec.utils._constants import (
    DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
    DEFAULT_ARTICLE_MODIFIED_TIMESTAMP_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_TOTAL_PAGEVIEWS_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_TOTAL_INVIEWS_COL,
    DEFAULT_IS_SUBSCRIBER_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_POSTCODE_COL,
    DEFAULT_GENDER_COL,
    DEFAULT_USER_COL,
    DEFAULT_AGE_COL,
)

from ebrec.utils._behaviors import truncate_history

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

df_history = (
    pl.scan_parquet(path.joinpath("test", "history.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
    .pipe(
        truncate_history,
        column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        history_size=10,
        padding_value=None,
        enable_warning=True,
    )
)


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
BERT_VECTOR = "bert_base_multilingual_cased"
CONTRASTIVE_VECTOR = "contrastive_vector"
ROBERTA_VECTOR = "xlm_roberta_base"
DOCUMENT_VECTOR = "document_vector"

emb_contrastive = pl.scan_parquet(
    path.parent.joinpath(
        f"embeddings/Ekstra_Bladet_contrastive_vector/{CONTRASTIVE_VECTOR}.parquet"
    )
)
emb_roberta = pl.scan_parquet(
    path.parent.joinpath(
        f"embeddings/FacebookAI_xlm_roberta_base/{ROBERTA_VECTOR}.parquet"
    )
)
emb_bert = pl.scan_parquet(
    path.parent.joinpath(
        f"embeddings/google_bert_base_multilingual_cased/{BERT_VECTOR}.parquet"
    )
)
emb_docvec = pl.scan_parquet(
    path.parent.joinpath(f"embeddings/Ekstra_Bladet_word2vec/{DOCUMENT_VECTOR}.parquet")
)
# =>
df_candidate_articles = (
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
df_candidate_articles.columns
# Make lookup-dictionary:
candidate_dict = {}
for row in df_candidate_articles.iter_rows(named=True):
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

PREDICTION_SCORE_COL = "prediction_score"


candidate_dict[list(candidate_dict)[0]].keys()


from ebrec.evaluation.beyond_accuracy import (
    IntralistDiversity,
    Distribution,
    Serendipity,
    Sentiment,
    Coverage,
    Novelty,
)

instralist_diversity = IntralistDiversity()
distribution = Distribution()
serendipity = Serendipity()
sentiment = Sentiment()
coverage = Coverage()
novelty = Novelty()


df_candidate_articles.select(DEFAULT_ARTICLE_ID_COL).sample(n=5)

df_ = make_prediction_score(
    df_candidate_articles,
    column=DEFAULT_TOTAL_INVIEWS_COL,
    desc=True,
    prediction_score_col=PREDICTION_SCORE_COL,
)
ranked_candidates = np.array(
    [df_.select(DEFAULT_ARTICLE_ID_COL).cast(pl.Utf8).to_series()]
)
TOP_N = 5
#
instralist_diversity(
    ranked_candidates[:, :TOP_N],
    lookup_dict=candidate_dict,
    lookup_key=CONTRASTIVE_VECTOR,
)

# Distributions:
distribution(
    ranked_candidates[:, :TOP_N],
    lookup_dict=candidate_dict,
    lookup_key="category_str",
)
distribution(
    ranked_candidates[:, :TOP_N],
    lookup_dict=candidate_dict,
    lookup_key="topics",
)


# Write a submission file:
prediction_score_lookup = create_lookup_dict(
    df_, DEFAULT_ARTICLE_ID_COL, PREDICTION_SCORE_COL
)

candidate_list_prediction_score = rank_predictions_by_score(
    [prediction_score_lookup.get(aid, 1e-6) for aid in candidate_list]
)

df_behaviors_ = df_behaviors.with_columns(
    pl.Series(DEFAULT_INVIEW_ARTICLES_COL, [candidate_list_prediction_score])
)

ids = df_behaviors["impression_id"].to_list()
preds = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL].to_list()

write_submission_file(
    impression_ids=ids,
    prediction_scores=preds,
    path=path_beyond.joinpath("predictions.txt"),
    filename_zip="predictions_inviews.zip",
)


def compute_transform_distribution(
    candidate_list: list[list[str]],
    lookup_dict: dict,
    lookup_key: str,
    suffix: str,
):
    # =>
    distribution = Distribution()
    return {
        **{"name": f"{distribution.name}{suffix}"},
        **distribution(
            candidate_list,
            lookup_dict=lookup_dict,
            lookup_key=lookup_key,
        ),
    }

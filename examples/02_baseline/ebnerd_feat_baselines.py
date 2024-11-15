from pathlib import Path
from tqdm import tqdm
import polars as pl

from ebrec.utils._python import (
    rank_predictions_by_score,
    write_submission_file,
    create_lookup_dict,
)
from ebrec.utils._constants import *

PATH = Path("~/ebnerd_data/ebnerd_testset")

df_behaviors = pl.scan_parquet(PATH.joinpath("test", "behaviors.parquet"))
df_articles = pl.scan_parquet(PATH.joinpath("articles.parquet"))

# ==== LOOKUP DICTS
clicked_dict = create_lookup_dict(
    df_articles.select(DEFAULT_ARTICLE_ID_COL, DEFAULT_TOTAL_PAGEVIEWS_COL).collect(),
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_TOTAL_PAGEVIEWS_COL,
)
inview_dict = create_lookup_dict(
    df_articles.select(DEFAULT_ARTICLE_ID_COL, DEFAULT_TOTAL_INVIEWS_COL).collect(),
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_TOTAL_INVIEWS_COL,
)
readtime_dict = create_lookup_dict(
    df_articles.select(DEFAULT_ARTICLE_ID_COL, DEFAULT_TOTAL_READ_TIME_COL).collect(),
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_TOTAL_READ_TIME_COL,
)

# Estimate:
df_inview_estimate = (
    df_behaviors.select(DEFAULT_INVIEW_ARTICLES_COL)
    .explode(DEFAULT_INVIEW_ARTICLES_COL)
    .select(pl.col(DEFAULT_INVIEW_ARTICLES_COL).value_counts())
    .unnest(DEFAULT_INVIEW_ARTICLES_COL)
    .collect()
)
inview_dict_estimate = create_lookup_dict(
    df_inview_estimate.select(DEFAULT_INVIEW_ARTICLES_COL, "count"),
    DEFAULT_INVIEW_ARTICLES_COL,
    "count",
)

# ==== CLICKED PREDICTIONS
CLICKED_SCORE_COL = "clicked_prediction_scores"
INVIEW_SCORE_COL = "inview_prediction_scores"
INVIEW_ESTIMATE_SCORE_COL = "inview_estimate_prediction_scores"
READTIME_SCORE_COL = "readtime_prediction_scores"

df_predictions = (
    df_behaviors.select(DEFAULT_IMPRESSION_ID_COL, DEFAULT_INVIEW_ARTICLES_COL)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element().replace(clicked_dict).fill_null(0))
        .alias(CLICKED_SCORE_COL)
    )
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element().replace(inview_dict).fill_null(0))
        .alias(INVIEW_SCORE_COL)
    )
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element().replace(inview_dict_estimate).fill_null(0))
        .alias(INVIEW_ESTIMATE_SCORE_COL)
    )
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element().replace(readtime_dict).fill_null(0))
        .alias(READTIME_SCORE_COL)
    )
    .collect()
)

# CONVERT TO RANKS:
impression_id = []
clicked_scores = []
inview_scores = []
inview_estimate_scores = []
readtime_scores = []
for row in tqdm(
    df_predictions.iter_rows(named=True),
    total=df_predictions.shape[0],
    ncols=80,
):
    impression_id.append(row[DEFAULT_IMPRESSION_ID_COL])
    clicked_scores.append(rank_predictions_by_score(row[CLICKED_SCORE_COL]))
    inview_scores.append(rank_predictions_by_score(row[INVIEW_SCORE_COL]))
    inview_estimate_scores.append(
        rank_predictions_by_score(row[INVIEW_ESTIMATE_SCORE_COL])
    )
    readtime_scores.append(rank_predictions_by_score(row[READTIME_SCORE_COL]))

#
for col, scores in zip(
    [
        CLICKED_SCORE_COL,
        INVIEW_SCORE_COL,
        INVIEW_ESTIMATE_SCORE_COL,
        READTIME_SCORE_COL,
    ],
    [clicked_scores, inview_scores, inview_estimate_scores, readtime_scores],
):
    print("Writing submission file for:", col)
    Path("downloads").mkdir(exist_ok=True)
    write_submission_file(
        impression_ids=impression_id,
        prediction_scores=scores,
        path="downloads/predictions.txt",
        filename_zip=f"{col}.zip",
    )

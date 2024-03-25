from ebrec.evaluation.beyond_accuracy import (
    IntralistDiversity,
    Distribution,
    Serendipity,
    Novelty,
)
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
from ebrec.utils._python import write_json_file, read_json_file
from pathlib import Path
import polars as pl
import numpy as np
import json

path = Path("examples/downloads/demo")
path_beyond = path.joinpath("baseline_prediction")
path_beyond.mkdir(exist_ok=True, parents=True)

#
df_behaviors = pl.scan_parquet(path.joinpath("test", "behaviors.parquet"))
df_articles = pl.scan_parquet(path.joinpath("articles.parquet"))
#
candidate_list = (
    df_behaviors.filter(pl.col("is_beyond_accuracy"))
    .select(pl.col(DEFAULT_INVIEW_ARTICLES_COL).first())
    .collect()
    .to_series()
)

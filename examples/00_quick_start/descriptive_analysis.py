from pathlib import Path
import polars as pl

from ebrec.utils._descriptive_analysis import (
    min_max_impression_time_behaviors,
    min_max_impression_time_history,
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._constants import *
from ebrec.utils._python import compute_npratio


def describe_text_column(
    df_articles: pl.DataFrame, column: str, decimals: int = 4
) -> dict[str, float]:
    """ """
    df_none_empty = df_articles.filter(pl.col(column).str.len_chars() > 0)[column]
    word_counts = df_none_empty.str.split(by=" ").list.len()
    character_counts = df_none_empty.str.lengths()
    return {
        f"{column}_word_count_mean": round(word_counts.mean(), decimals),
        f"{column}_word_count_std": round(word_counts.std(), decimals),
        f"{column}_character_count_mean": round(character_counts.mean(), decimals),
        f"{column}_character_count_std": round(character_counts.std(), decimals),
    }


PATH = Path("~/ebnerd_data")
TRAIN_VAL_SPLIT = f"ebnerd_demo"  # [ebnerd_demo, ebnerd_small, ebnerd_large]
TEST_SPLIT = f"ebnerd_testset"

df_behaviors_train = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "train", "behaviors.parquet")
)
df_history_train = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "train", "history.parquet")
)
df_behaviors_val = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "validation", "behaviors.parquet")
)
df_history_val = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TRAIN_VAL_SPLIT, "validation", "history.parquet")
)
df_behaviors_test = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TEST_SPLIT, "test", "behaviors.parquet")
)
df_history_test = df_behaviors = pl.scan_parquet(
    PATH.joinpath(TEST_SPLIT, "test", "history.parquet")
)
#
df_behaviors_concat = pl.concat(
    [df_behaviors_train, df_behaviors_val, df_behaviors_test]
)
df_history_concat = pl.concat([df_history_train, df_history_val, df_history_test])
#
df_articles = pl.scan_parquet(PATH.joinpath(TEST_SPLIT, "articles.parquet"))

# UNIQUE USERS:
n_users = int(df_behaviors_concat.select(DEFAULT_USER_COL).unique().collect().shape[0])

# N-IMPRESSSIONS:
n_impressions = df_behaviors_concat.select(DEFAULT_USER_COL).collect().shape[0]

# UNIQUE ARTICLES:
n_articles = df_articles.select(DEFAULT_ARTICLE_ID_COL).unique().collect().shape[0]

# ARTICLE CATEGORIES:
n_categories = df_articles.select(DEFAULT_CATEGORY_COL).unique().collect().shape[0]
n_subcategories = (
    df_articles.select(pl.col(DEFAULT_SUBCATEGORY_COL).explode())
    .unique()
    .collect()
    .shape[0]
)

# NPRATIO:
pos = (
    pl.concat([df_behaviors_train, df_behaviors_val])
    .select(pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len())
    .sum()
    .collect()
)[DEFAULT_CLICKED_ARTICLES_COL][0]
neg = (
    pl.concat([df_behaviors_train, df_behaviors_val])
    .select(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len()
        - pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len()
    )
    .sum()
    .collect()
)[DEFAULT_INVIEW_ARTICLES_COL][0]
npratio = compute_npratio(n_pos=pos, n_neg=neg)

# DESICRIBE TEXT COLUMNS:
title_desc = describe_text_column(df_articles.collect(), DEFAULT_TITLE_COL, decimals=2)
subtitle_desc = describe_text_column(
    df_articles.collect(), DEFAULT_SUBTITLE_COL, decimals=2
)
body_desc = describe_text_column(df_articles.collect(), DEFAULT_BODY_COL, decimals=2)

# History
hist_mean = (
    df_history_concat.select(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.len())
    .mean()
    .collect()[DEFAULT_HISTORY_ARTICLE_ID_COL][0]
)
hist_std = (
    df_history_concat.select(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.len())
    .std()
    .collect()[DEFAULT_HISTORY_ARTICLE_ID_COL][0]
)


# PRINT:
print(f"# Users: {n_users}")
print(f"# Impressions: {n_impressions}")
print(f"# Articles: {n_articles}")
print(f"# Categories: {n_categories}")
print(f"# Subcategories: {n_subcategories}")
print(f"Avg. NP-raio: {npratio}")
print(f"Avg. Impression per user: {n_impressions/n_users}")
print(
    f"Avg. title len. (words): {title_desc[list(title_desc)[0]]} \pm {title_desc[list(title_desc)[1]]}"
)
print(
    f"Avg. title len. (words): {subtitle_desc[list(subtitle_desc)[0]]} \pm {subtitle_desc[list(subtitle_desc)[0]]}"
)
print(
    f"Avg. title len. (words): {body_desc[list(body_desc)[0]]} \pm {body_desc[list(body_desc)[1]]}"
)
print("Avg. history len.: {:.2f} \pm {:.2f}".format(hist_mean, hist_std))

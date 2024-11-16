from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os

from ebrec.utils._constants import *

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes, split_df
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel


def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# conda activate ./venv/; python examples/00_quick_start/nrms_ebnerd_small.py

PATH = Path("~/ebnerd_data").expanduser()
DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
DT_NOW = dt.datetime.now()

MODEL_NAME = f"NRMS-{DT_NOW}"
MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")

DATASPLIT = "ebnerd_small"
FRACTION = 1.0
EPOCHS = 5

TEST_SAMPLES = 100_000

MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 20
NPRATIO = 4

hparams_nrms.title_size = MAX_TITLE_LENGTH
hparams_nrms.history_size = HISTORY_SIZE

hparams_nrms.head_num = 20
hparams_nrms.head_dim = 20
hparams_nrms.attention_hidden_dim = 200

hparams_nrms.optimizer = "adam"
hparams_nrms.loss = "cross_entropy_loss"
hparams_nrms.dropout = 0.35
hparams_nrms.learning_rate = 1e-5

WITH_REPLACEMENT = True
MIN_VALS = 0

BS_TRAIN = 32
BS_TEST = 32

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION)
    .select(COLUMNS)
    .filter(
        (
            pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len()
            - pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len()
        )
        > MIN_VALS
    )
    .pipe(
        sampling_strategy_wu2019,
        npratio=NPRATIO,
        shuffle=True,
        with_replacement=WITH_REPLACEMENT,
        seed=123,
    )
    .pipe(create_binary_labels_column)
)

# =>
df_val = ebnerd_from_path(
    PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE
).select(COLUMNS)[:TEST_SAMPLES]

df_val, df_test = split_df(df_val, fraction=0.5, seed=123)

df_val = (
    df_val.filter(
        (
            pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len()
            - pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len()
        )
        > MIN_VALS
    )
    .pipe(
        sampling_strategy_wu2019,
        npratio=NPRATIO,
        shuffle=True,
        with_replacement=WITH_REPLACEMENT,
        seed=123,
    )
    .pipe(create_binary_labels_column)
)

df_test = df_test.pipe(create_binary_labels_column, shuffle=False)

# =>
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
TRANSFORMER_MODEL_NAME = "Maltehb/danish-bert-botxo"

TEXT_COLUMNS_TO_USE = [DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL, DEFAULT_BODY_COL]
article_path = PATH.joinpath(
    f"articles_{TRANSFORMER_MODEL_NAME.replace('/', '_')}_{MAX_TITLE_LENGTH}.parquet"
)
# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
if not article_path.exists():
    df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
    )
    df_articles.write_parquet(article_path)
else:
    df_articles = pl.read_parquet(article_path)
    token_col_title = df_articles.columns[-1]

# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)

# =>
train_dataloader = NRMSDataLoaderPretransform(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BS_TRAIN,
)

val_dataloader = NRMSDataLoaderPretransform(
    behaviors=df_val,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BS_TEST,
)

test_dataloader = NRMSDataLoaderPretransform(
    behaviors=df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BS_TEST,
)

train_dataloader_test = NRMSDataLoaderPretransform(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BS_TEST,
)

# CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=20, restore_best_weights=True
)
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", mode="min", factor=0.5, patience=2, min_lr=1e-6
)
callbacks = [lr_scheduler, early_stopping, modelcheckpoint, tensorboard_callback]

model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)
model.model.compile(
    optimizer=model.model.optimizer,
    loss="categorical_crossentropy",
    metrics=["AUC"],
)

hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=15,
    callbacks=callbacks,
)
print("loading model...")
model.model.load_weights(MODEL_WEIGHTS)


pred_test = model.scorer.predict(train_dataloader_test)
print("Adding prediction-scores...")
df_train_test = add_prediction_scores(df_train, pred_test.tolist())

print("Evaluating...")
metrics = MetricEvaluator(
    labels=df_train_test["labels"].to_list(),
    predictions=df_train_test["scores"].to_list(),
    metric_functions=[
        AucScore(),
        # MrrScore(),
        # NdcgScore(k=5),
        # NdcgScore(k=10),
    ],
)
metrics.evaluate()
print(metrics.evaluations)
# 0.7383
# =>
pred_test = model.scorer.predict(test_dataloader)
print("Adding prediction-scores...")
df_test = add_prediction_scores(df_test, pred_test.tolist())

print("Evaluating...")
metrics = MetricEvaluator(
    labels=df_test["labels"].to_list(),
    predictions=df_test["scores"].to_list(),
    metric_functions=[
        AucScore(),
        # MrrScore(),
        # NdcgScore(k=5),
        # NdcgScore(k=10),
    ],
)
metrics.evaluate()
print(metrics.evaluations)

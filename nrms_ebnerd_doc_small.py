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
from ebrec.models.newsrec.model_config import (
    hparams_nrms,
    hparams_nrms_docvec,
    print_hparams,
)
from ebrec.models.newsrec.nrms_docvec import NRMSModel_docvec
from ebrec.models.newsrec import NRMSModel

from utils import ebnerd_from_path, PATH, COLUMNS, DUMP_DIR, down_sample_on_users

# conda activate ./venv/; python nrms_ebnerd_doc.py
# conda activate ./venv/; tensorboard --logdir=ebnerd_predictions/runs

model_func = NRMSModel_docvec
DT_NOW = dt.datetime.now()
SEED = 123

MODEL_NAME = f"{model_func.__name__}-{DT_NOW}"
MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")

DATASPLIT = "ebnerd_small"
BS_TRAIN = 32
BS_TEST = 32

TEST_SAMPLES = 100_000
EPOCHS = 10

MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 20
NPRATIO = 4

TRAIN_FRACTION = 0.2
WITH_REPLACEMENT = True
MIN_N_INVIEWS = 0  # 0 = all
MAX_N_IMPR_USERS = 0  # 0 = all

hparams_nrms_docvec.title_size = 768
hparams_nrms_docvec.history_size = HISTORY_SIZE
# MODEL ARCHITECTURE
hparams_nrms_docvec.head_num = 16
hparams_nrms_docvec.head_dim = 16
hparams_nrms_docvec.attention_hidden_dim = 200
# MODEL OPTIMIZER:
hparams_nrms_docvec.optimizer = "adam"
hparams_nrms_docvec.loss = "cross_entropy_loss"
hparams_nrms_docvec.dropout = 0.2
hparams_nrms_docvec.learning_rate = 1e-4
hparams_nrms_docvec.newsencoder_l2_regularization = 1e-4
hparams_nrms_docvec.newsencoder_units_per_layer = [256, 256, 256]

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .sample(fraction=TRAIN_FRACTION)
    .select(COLUMNS)
    .filter(
        (
            pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len()
            - pl.col(DEFAULT_CLICKED_ARTICLES_COL).list.len()
        )
        > MIN_N_INVIEWS
    )
    .pipe(down_sample_on_users, n=MAX_N_IMPR_USERS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=NPRATIO,
        shuffle=True,
        with_replacement=WITH_REPLACEMENT,
        seed=SEED,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=1.0, seed=SEED, shuffle=True)
)
print(f"Train-samples: {df_train.height}")

# =>
df_val = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .sample(n=TEST_SAMPLES, seed=123)
)
df_val, df_test = split_df(df_val, fraction=0.5, seed=123)

df_val = df_val.pipe(
    sampling_strategy_wu2019,
    npratio=NPRATIO,
    shuffle=True,
    with_replacement=WITH_REPLACEMENT,
    seed=123,
).pipe(create_binary_labels_column)

df_test = df_test.pipe(create_binary_labels_column, shuffle=False)

# =>
df_articles = pl.read_parquet(
    PATH.joinpath(
        "artifacts/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet"
    )
)

article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
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

# CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=4, restore_best_weights=True
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
    monitor="val_auc", mode="max", factor=0.2, patience=2, min_lr=1e-6
)
callbacks = [lr_scheduler, early_stopping, modelcheckpoint, tensorboard_callback]

model = model_func(
    hparams=hparams_nrms_docvec,
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
    epochs=EPOCHS,
    callbacks=callbacks,
)
print("loading model...")
model.model.load_weights(MODEL_WEIGHTS)

# ===
# train_dataloader_test = NRMSDataLoaderPretransform(
#     behaviors=df_train,
#     article_dict=article_mapping,
#     unknown_representation="zeros",
#     history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
#     eval_mode=True,
#     batch_size=BS_TEST,
# )
# pred_test = model.scorer.predict(train_dataloader_test)
# print("Adding prediction-scores...")
# df_train_test = add_prediction_scores(df_train, pred_test.tolist())

# print("Evaluating...")
# metrics = MetricEvaluator(
#     labels=df_train_test["labels"].to_list(),
#     predictions=df_train_test["scores"].to_list(),
#     metric_functions=[AucScore()],
# )
# metrics.evaluate()
# print(metrics.evaluations)
# 0.7383
# =>
test_dataloader = NRMSDataLoader(
    behaviors=df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=32,
)

pred_test = model.scorer.predict(test_dataloader)
print("Adding prediction-scores...")
df_test = add_prediction_scores(df_test, pred_test.tolist())

print("Evaluating...")
metrics = MetricEvaluator(
    labels=df_test["labels"].to_list(),
    predictions=df_test["scores"].to_list(),
    metric_functions=[AucScore()],
)
metrics.evaluate()
print_hparams(hparams_nrms_docvec)
print(metrics.evaluations)

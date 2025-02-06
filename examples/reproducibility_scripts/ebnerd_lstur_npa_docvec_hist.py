from transformers import AutoTokenizer, AutoModel
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._articles import convert_text2encoding_with_transformers

from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import shutil
import gc
import os

from ebrec.utils._constants import *

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    truncate_history,
    ebnerd_from_path,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

from ebrec.utils._python import (
    rank_predictions_by_score,
    write_submission_file,
    create_lookup_dict,
    write_json_file,
)
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._polars import split_df_chunks, concat_str_columns

from ebrec.models.newsrec.dataloader import LSTURDataLoader
from ebrec.models.newsrec.model_config import (
    hparams_lstur_docvec,
    hparams_npa_docvec,
    hparams_to_dict,
    print_hparams,
)
from ebrec.models.newsrec.lstur_docvec import LSTURDocVec
from ebrec.models.newsrec.npa_docvec import NPADocVec

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from arguments.args_lstur_docvec import get_args as get_args_lstur
from arguments.args_npa_docvec import get_args as get_args_npa

args = get_args_lstur()
args = get_args_lstur() if args.model == "LSTURDocVec" else get_args_npa()

for arg, val in vars(args).items():
    print(f"{arg} : {val}")

# conda activate ./venv; python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_hist.py

PATH = Path(args.data_path).expanduser()
# Access arguments as variables
SEED = args.seed
DATASPLIT = args.datasplit
DEBUG = args.debug
BS_TRAIN = args.bs_train
BS_TEST = args.bs_test
BATCH_SIZE_TEST_WO_B = args.batch_size_test_wo_b
BATCH_SIZE_TEST_W_B = args.batch_size_test_w_b
HISTORY_SIZE = args.history_size
NPRATIO = args.npratio
EPOCHS = args.epochs
TRAIN_FRACTION = args.train_fraction if not DEBUG else 0.0001
FRACTION_TEST = args.fraction_test if not DEBUG else 0.0001


# =====================================================================================
#  ############################# UNIQUE FOR NRMSModel ################################
# =====================================================================================

# Data-path
DOC_VEC_PATH = PATH.joinpath(f"artifacts/{args.document_embeddings}")
print("Initiating articles...")
df_articles = pl.read_parquet(DOC_VEC_PATH)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
)

# Model in use:
model_func = LSTURDocVec if args.model == "LSTURDocVec" else NPADocVec
hparams = (
    hparams_lstur_docvec if model_func.__name__ == "LSTURDocVec" else hparams_npa_docvec
)
#
for key, value in vars(args).items():
    if hasattr(hparams, key):
        setattr(hparams, key, value)

print_hparams(hparams)

# =====================================================================================
#  ############################# UNIQUE FOR NRMSDocVec ###############################
# =====================================================================================


# Dump paths:
DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
#
DT_NOW = dt.datetime.now()
#
MODEL_NAME = model_func.__name__
MODEL_OUTPUT_NAME = f"{MODEL_NAME}-{DT_NOW}"
#
ARTIFACT_DIR = DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
# Model monitoring:
MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_OUTPUT_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_OUTPUT_NAME}")
# Just trying keeping the dataframe slime:
COLUMNS = [
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
]
# Store hparams
write_json_file(
    hparams_to_dict(hparams),
    ARTIFACT_DIR.joinpath(f"{MODEL_NAME}_hparams.json"),
)
write_json_file(vars(args), ARTIFACT_DIR.joinpath(f"{MODEL_NAME}_argparser.json"))

# =====================================================================================

df = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .sample(fraction=TRAIN_FRACTION, shuffle=True, seed=SEED)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=SEED,
    )
    .pipe(create_binary_labels_column)
)
#
last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)

user_id_mapping = create_lookup_dict(
    df_train.select(DEFAULT_USER_COL)
    .unique()
    .with_row_index(name="id", offset=1)[: args.n_users],
    key=DEFAULT_USER_COL,
    value="id",
)
hparams.n_users = len(user_id_mapping)

# =====================================================================================
print(f"Initiating training-dataloader")
train_dataloader = LSTURDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    user_id_mapping=user_id_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BS_TRAIN,
)

val_dataloader = LSTURDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    user_id_mapping=user_id_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BS_TRAIN,
)

# =====================================================================================
# CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=4,
    restore_best_weights=True,
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
    monitor="val_auc",
    mode="max",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
)
callbacks = [tensorboard_callback, early_stopping, modelcheckpoint, lr_scheduler]

# =====================================================================================
model = model_func(
    hparams=hparams,
    seed=42,
)
model.model.compile(
    optimizer=model.model.optimizer,
    loss=model.model.loss,
    metrics=["AUC"],
)
f"Initiating {MODEL_NAME}, start training..."
# =>
hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print(f"loading model: {MODEL_WEIGHTS}")
model.model.load_weights(MODEL_WEIGHTS)

# =====================================================================================

# First filter: only keep users with >FILTER_MIN_HISTORY in history-size
FILTER_MIN_HISTORY = 100

# =>
df = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "validation"),
        history_size=200,  # > FILTER_MIN_HISTORY
        padding=None,  # NO PADDING!
    )
    .sample(fraction=FRACTION_TEST)
    .filter(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.len() >= FILTER_MIN_HISTORY)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
)

pairs = [
    (1, 256),
    (2, 256),
    # (3, 256),
    (4, 256),
    # (5, 256),
    # (6, 256),
    # (7, 256),
    (8, 256),
    # (9, 256),
    (10, 256),
    # (16, 128),
    (20, 128),
    (40, 64),
    (80, 64),
    (FILTER_MIN_HISTORY, 8),
]

aucs = []
hists = []
for hist_size, batch_size in pairs:
    print(f"History size: {hist_size}, Batch size: {batch_size}")

    df_ = df.pipe(
        truncate_history,
        column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        history_size=hist_size,
        padding_value=0,
        enable_warning=False,
    )

    test_dataloader = LSTURDataLoader(
        behaviors=df_,
        article_dict=article_mapping,
        user_id_mapping=user_id_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=batch_size,
    )

    scores = model.scorer.predict(test_dataloader)

    df_pred = add_prediction_scores(df_, scores.tolist())

    metrics = MetricEvaluator(
        labels=df_pred["labels"],
        predictions=df_pred["scores"],
        metric_functions=[AucScore()],
    )
    metrics.evaluate()
    auc = metrics.evaluations["auc"]
    aucs.append(round(auc, 4))
    hists.append(hist_size)
    print(f"{auc} (History size: {hist_size}, Batch size: {batch_size})")

for h, a in zip(hists, aucs):
    print(f"({a}, {h}),")

results = {h: a for h, a in zip(hists, aucs)}
write_json_file(results, ARTIFACT_DIR.joinpath("auc_history_length.json"))

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
    add_prediction_scores,
    ebnerd_from_path,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

from ebrec.utils._python import write_submission_file, rank_predictions_by_score
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._polars import split_df_chunks

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms_docvec
from ebrec.models.newsrec.nrms_docvec import NRMSModel_docvec

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# conda activate ./venv/; python nrms_ebnerd_doc_large.py
# conda activate ./venv/; tensorboard --logdir=ebnerd_predictions/runs

# =====================================================================================
# Model in use:
model_func = NRMSModel_docvec
# Data-path
PATH = Path("~/ebnerd_data").expanduser()
DOC_VEC_PATH = PATH.joinpath(
    "artifacts/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet"
)
# Dump paths:
DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
#
DT_NOW = dt.datetime.now()
SEED = 123
#
MODEL_NAME = f"{model_func.__name__}-{DT_NOW}"
# Model monitoring:
MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")
# Evaluating the test test can be memory intensive, we'll chunk it up:
TEST_CHUNKS_DUMP = DUMP_DIR.joinpath("test_predictions", MODEL_NAME, "test_chunks")
TEST_CHUNKS_DUMP.mkdir(parents=True, exist_ok=True)
N_CHUNKS_TEST = 10
CHUNKS_DONE = 0  # if it crashes, you can start from here.
# Just trying keeping the dataframe slime:
COLUMNS = [
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
]
# =====================================================================================

DATASPLIT = "ebnerd_small"
BS_TRAIN = 32
BS_TEST = 32
BATCH_SIZE_TEST_WO_B = 32
BATCH_SIZE_TEST_W_B = 4

# - NRMSDataLoaderPretransform: speed efficient.
# - NRMSDataLoader: memory efficient.
NRMSLoader_training = NRMSDataLoaderPretransform  # NRMSDataLoader

HISTORY_SIZE = 20
NPRATIO = 4

EPOCHS = 5
TRAIN_FRACTION = 1.0
FRACTION_TEST = 1.0

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

# =====================================================================================
# We'll use the training + validation sets for training.
df = (
    pl.concat(
        [
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE
            ),
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE
            ),
        ]
    )
    .sample(fraction=TRAIN_FRACTION, shuffle=True, seed=SEED)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=NPRATIO,
        shuffle=True,
        with_replacement=True,
        seed=SEED,
    )
    .pipe(create_binary_labels_column)
)

# We keep the last day of our training data as the validation set.
last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)

# =====================================================================================
df_articles = pl.read_parquet(DOC_VEC_PATH)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
)

# =====================================================================================
print(f"Initiating training-dataloader")
train_dataloader = NRMSLoader_training(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BS_TRAIN,
)

val_dataloader = NRMSLoader_training(
    behaviors=df_validation,
    article_dict=article_mapping,
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
print(f"Initiating {model_func.__class__.__class__}, start training...")
model = model_func(
    hparams=hparams_nrms_docvec,
    seed=42,
)
model.model.compile(
    optimizer=model.model.optimizer,
    loss=model.model.loss,
    metrics=["AUC"],
)
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
# =====================================================================================
print("Initiating testset...")
df_test = (
    ebnerd_from_path(PATH.joinpath("ebnerd_testset", "test"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION_TEST)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.first()
        .alias(DEFAULT_CLICKED_ARTICLES_COL)
    )
    .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element() * 0)
        .alias(DEFAULT_LABELS_COL)
    )
)
# Split test in beyond-accuracy TRUE / FALSE. In the BA 'article_ids_inview' is 250.
df_test_wo_beyond = df_test.filter(~pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
df_test_w_beyond = df_test.filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))

df_test_chunks = split_df_chunks(df_test_wo_beyond, n_chunks=N_CHUNKS_TEST)
df_pred_test_wo_beyond = []
print("Initiating testset without beyond-accuracy...")
for i, df_test_chunk in enumerate(df_test_chunks[CHUNKS_DONE:], start=1 + CHUNKS_DONE):
    print(f"Test chunk: {i}/{len(df_test_chunks)}")
    # Initialize DataLoader
    test_dataloader_wo_b = NRMSDataLoader(
        behaviors=df_test_chunk,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=BATCH_SIZE_TEST_WO_B,
    )
    # Predict and clear session
    scores = model.scorer.predict(test_dataloader_wo_b)
    tf.keras.backend.clear_session()

    # Process the predictions
    df_test_chunk = add_prediction_scores(df_test_chunk, scores.tolist()).with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )

    # Save the processed chunk
    df_test_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_CHUNKS_DUMP.joinpath(f"pred_wo_ba_{i}.parquet")
    )

    # Append and clean up
    df_pred_test_wo_beyond.append(df_test_chunk)

    # Cleanup
    del df_test_chunk, test_dataloader_wo_b, scores
    gc.collect()

df_pred_test_wo_beyond = pl.concat(df_pred_test_wo_beyond)
df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DUMP.joinpath("pred_wo_ba.parquet")
)
# =====================================================================================
print("Initiating testset with beyond-accuracy...")
test_dataloader_w_b = NRMSDataLoader(
    behaviors=df_test_w_beyond,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE_TEST_W_B,
)
scores = model.scorer.predict(test_dataloader_w_b)
df_pred_test_w_beyond = add_prediction_scores(
    df_test_w_beyond, scores.tolist()
).with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)
df_pred_test_w_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DUMP.joinpath("pred_w_ba.parquet")
)

# =====================================================================================
print("Saving prediction results...")
df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])
df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DUMP.parent.joinpath("test_predictions.parquet")
)
write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=DUMP_DIR.joinpath("predictions.txt"),
    filename_zip=f"{DATASPLIT}_predictions-{MODEL_NAME}.zip",
)

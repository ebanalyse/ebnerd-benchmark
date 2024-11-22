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
    write_submission_file,
    rank_predictions_by_score,
    write_json_file,
)
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._polars import split_df_chunks, concat_str_columns

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import (
    hparams_nrms,
    hparams_nrms_docvec,
    hparams_to_dict,
    print_hparams,
)
from ebrec.models.newsrec.nrms_docvec import NRMSDocVec
from ebrec.models.newsrec import NRMSModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from args_nrms_docvec import get_args

args = get_args()

for arg, val in vars(args).items():
    print(f"{arg} : {val}")

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

NRMSLoader_training = (
    NRMSDataLoaderPretransform
    if args.nrms_loader == "NRMSDataLoaderPretransform"
    else NRMSDataLoader
)

# =====================================================================================
#  ############################# UNIQUE FOR NRMSModel ################################
# =====================================================================================

# Model in use:
model_func = NRMSDocVec
hparams = hparams_nrms_docvec
#
hparams.title_size = args.title_size
hparams.history_size = args.history_size
hparams.head_num = args.head_num
hparams.head_dim = args.head_dim
hparams.attention_hidden_dim = args.attention_hidden_dim
hparams.newsencoder_units_per_layer = args.newsencoder_units_per_layer
hparams.optimizer = args.optimizer
hparams.loss = args.loss
hparams.dropout = args.dropout
hparams.learning_rate = args.learning_rate
hparams.newsencoder_l2_regularization = args.newsencoder_l2_regularization


# =============
# Data-path
DOC_VEC_PATH = PATH.joinpath(f"artifacts/{args.document_embeddings}")
print("Initiating articles...")
df_articles = pl.read_parquet(DOC_VEC_PATH)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
)

# =====================================================================================
#  ############################# UNIQUE FOR NRMSDocVec ###############################
# =====================================================================================

print_hparams(hparams)

# Dump paths:
DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
#
DT_NOW = dt.datetime.now()
#
MODEL_NAME = model_func.__name__
MODEL_OUTPUT_NAME = f"{MODEL_NAME}-{DT_NOW}"
#
ARTIFACT_DIR = DUMP_DIR.joinpath("test_predictions", MODEL_OUTPUT_NAME)
# Model monitoring:
MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_OUTPUT_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_OUTPUT_NAME}")
# Evaluating the test test can be memory intensive, we'll chunk it up:
TEST_CHUNKS_DIR = ARTIFACT_DIR.joinpath("test_chunks")
TEST_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
N_CHUNKS_TEST = args.n_chunks_test
CHUNKS_DONE = args.chunks_done  # if it crashes, you can start from here.
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
# We'll use the training + validation sets for training.
df = (
    pl.concat(
        [
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "train"),
                history_size=HISTORY_SIZE,
                padding=0,
            ),
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "validation"),
                history_size=HISTORY_SIZE,
                padding=0,
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
print("Initiating testset...")
df_test = (
    ebnerd_from_path(
        PATH.joinpath("ebnerd_testset", "test"),
        history_size=HISTORY_SIZE,
        padding=0,
    )
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
        TEST_CHUNKS_DIR.joinpath(f"pred_wo_ba_{i}.parquet")
    )

    # Append and clean up
    df_pred_test_wo_beyond.append(df_test_chunk)

    # Cleanup
    del df_test_chunk, test_dataloader_wo_b, scores
    gc.collect()

df_pred_test_wo_beyond = pl.concat(df_pred_test_wo_beyond)
df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DIR.joinpath("pred_wo_ba.parquet")
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
    TEST_CHUNKS_DIR.joinpath("pred_w_ba.parquet")
)

# =====================================================================================
print("Saving prediction results...")
df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])
df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    ARTIFACT_DIR.joinpath("test_predictions.parquet")
)

if TEST_CHUNKS_DIR.exists() and TEST_CHUNKS_DIR.is_dir():
    shutil.rmtree(TEST_CHUNKS_DIR)

write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=ARTIFACT_DIR.joinpath("predictions.txt"),
    filename_zip=f"{MODEL_NAME}-{SEED}-{DATASPLIT}.zip",
)

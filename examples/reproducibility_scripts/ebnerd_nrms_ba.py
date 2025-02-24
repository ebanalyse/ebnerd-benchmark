from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
import os

from ebrec.utils._constants import *

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    ebnerd_from_path,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._python import write_json_file, get_top_n_candidates, split_array

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import (
    hparams_nrms_docvec,
    hparams_to_dict,
    print_hparams,
)
from ebrec.models.newsrec.nrms_docvec import NRMSDocVec

from ebrec.evaluation import (
    IntralistDiversity,
    Distribution,
    Serendipity,
    Coverage,
    Novelty,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# conda activate ./venv; python -i examples/reproducibility_scripts/ebnerd_nrms_ba.py --title_size 768 --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet --debug

from arguments.args_nrms_docvec import get_args

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

# Data-path
DOC_VEC_PATH = PATH.joinpath(f"artifacts/{args.document_embeddings}")
DOC_VEC_PATH_BA = PATH.joinpath(
    "artifacts/Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet"
)
print("Initiating articles...")
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
df_emb = pl.read_parquet(DOC_VEC_PATH)
df_emb_ba = pl.read_parquet(DOC_VEC_PATH_BA)

df_articles = df_articles.join(df_emb, on=DEFAULT_ARTICLE_ID_COL).join(
    df_emb_ba, on=DEFAULT_ARTICLE_ID_COL
)

article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_emb.columns[-1]
)

# Model in use:
model_func = NRMSDocVec
hparams = hparams_nrms_docvec
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
emb_name = args.document_embeddings.split("/")[1].split(".")[0]
MODEL_NAME = model_func.__name__
MODEL_OUTPUT_NAME = f"{MODEL_NAME}-{emb_name}-{DT_NOW}"
#
ARTIFACT_DIR = DUMP_DIR.joinpath("test_predictions", MODEL_OUTPUT_NAME)
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
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "train"),
        history_size=HISTORY_SIZE,
        padding=0,
    )
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

# =>
df_ = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "validation"),
        history_size=HISTORY_SIZE,
        padding=0,
    )
    .sample(fraction=FRACTION_TEST)
    .pipe(create_binary_labels_column)
)

test_dataloader = NRMSDataLoader(
    behaviors=df_,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BS_TEST,
)

scores = model.scorer.predict(test_dataloader)

df_pred = add_prediction_scores(df_, scores.tolist())
metrics = MetricEvaluator(
    labels=df_pred["labels"],
    predictions=df_pred["scores"],
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
results = metrics.evaluate()
print(results.evaluations)
write_json_file(results.evaluations, ARTIFACT_DIR.joinpath("results.json"))

del (
    test_dataloader,
    train_dataloader,
    val_dataloader,
    df_,
    df_train,
    df_validation,
    df_pred,
)

# BA results:
print("Initiating testset...")
df_ba = (
    ebnerd_from_path(
        PATH.joinpath("ebnerd_testset", "test"),
        history_size=HISTORY_SIZE,
        padding=0,
    )
    .filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
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

DEFAULT_TOTAL_PAGEVIEWS_COL_NORMALIZED_MAX = (
    DEFAULT_TOTAL_PAGEVIEWS_COL + "_normalized_max"
)
DEFAULT_TOTAL_PAGEVIEWS_COL_NORMALIZED_MIN_MAX = (
    DEFAULT_TOTAL_PAGEVIEWS_COL + "_normalized_min_max"
)

MIN_X = df_articles[DEFAULT_TOTAL_PAGEVIEWS_COL].min()
MAX_X = df_articles[DEFAULT_TOTAL_PAGEVIEWS_COL].max()
MIN_RANGE = 1e-4
MAX_RANGE = 1.0

df_articles = (
    df_articles.with_columns(
        pl.col(DEFAULT_TOTAL_INVIEWS_COL, DEFAULT_TOTAL_PAGEVIEWS_COL).fill_null(1)
    )
    .with_columns(
        (  # SIMPLE MAX NORMALIZATION: x / max()
            pl.col(DEFAULT_TOTAL_PAGEVIEWS_COL)
            / pl.col(DEFAULT_TOTAL_PAGEVIEWS_COL).max()
        ).alias(DEFAULT_TOTAL_PAGEVIEWS_COL_NORMALIZED_MAX)
    )
    .with_columns(
        (  #  MIN-MAX NORMALIZATION: ( x_i − X_min ⁡ ) / ( X_max ⁡ − X_min ⁡ ) * (max_range − min_range) + min_range
            ((pl.col(DEFAULT_TOTAL_PAGEVIEWS_COL) - MIN_X) / (MAX_X - MIN_X))
            * (MAX_RANGE - MIN_RANGE)
            + MIN_RANGE
        ).alias(
            DEFAULT_TOTAL_PAGEVIEWS_COL_NORMALIZED_MIN_MAX
        )
    )
)


ba_dataloader = NRMSDataLoader(
    behaviors=df_ba,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=4,
)
ba_scores = model.scorer.predict(ba_dataloader)
_ba_scores = split_array(ba_scores.ravel(), 250)
df_ba = df_ba.with_columns(pl.Series("scores", _ba_scores))
ba_emb_name = df_emb_ba.columns[-1]

articles_dict = {}
for row in df_articles.iter_rows(named=True):
    articles_dict[row[DEFAULT_ARTICLE_ID_COL]] = row

topn = 5
recs_scores = np.array(df_ba.select(pl.col("scores")).to_series().to_list())
cand_list = np.array(df_ba[DEFAULT_INVIEW_ARTICLES_COL][0])
hist = df_ba.select(DEFAULT_HISTORY_ARTICLE_ID_COL).to_series().to_list()
recs_topn = get_top_n_candidates(
    candidates_array=cand_list, scores_matrix=recs_scores, top_n=topn
)

# =>
intralist_diversity = IntralistDiversity()
distribution = Distribution()
serendipity = Serendipity()
coverage = Coverage()
novelty = Novelty()

div = intralist_diversity(
    R=recs_topn, lookup_dict=articles_dict, lookup_key=ba_emb_name
).mean()
dist_sent = distribution(
    R=recs_topn, lookup_dict=articles_dict, lookup_key="sentiment_label"
)
dist_cat = distribution(
    R=recs_topn, lookup_dict=articles_dict, lookup_key="category_str"
)
ser = serendipity(
    R=recs_topn, H=hist, lookup_dict=articles_dict, lookup_key=ba_emb_name
).mean()
cov = coverage(R=recs_topn, C=cand_list)[1]
nov = novelty(
    R=recs_topn,
    lookup_dict=articles_dict,
    lookup_key=DEFAULT_TOTAL_PAGEVIEWS_COL_NORMALIZED_MIN_MAX,
).mean()

results.evaluations.update(
    {"diversity": div, "serendipity": ser, "coverage": cov, "novelty": nov}
)
results.evaluations.update(dist_sent)
results.evaluations.update(dist_cat)
write_json_file(results.evaluations, ARTIFACT_DIR.joinpath("results.json"))

from optuna.trial import TrialState
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import datetime as dt
from tqdm import tqdm
import polars as pl
import numpy as np
import optuna
import mlflow

from ebrec.evaluation.metrics import roc_auc_score

# ===
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch import nn

from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._python import write_json_file

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    truncate_history,
    ebnerd_from_path,
)

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import (
    hparams_nrms_docvec,
    hparams_to_dict,
    print_hparams,
)

from ebrec.models.newsrec.nrms_docvec import NRMSDocVec
from ebrec.utils._constants import *

#
hparams = hparams_nrms_docvec
model_func = NRMSDocVec
#
DEBUG = False
SEED = 123
# ====>
DATASPLIT = "ebnerd_small"
PATH = Path("~/ebnerd_data").expanduser()
# Dump paths:
DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
#
DT_NOW = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

MODEL_NAME = model_func.__name__
MODEL_OUTPUT_NAME = f"{MODEL_NAME}-{DT_NOW}"

# ==
N_TRIALS = 25
#
BS_TRAIN = 32
BS_TEST = 32
EPOCHS = 10

# HPARAM.
HISTORY_SIZE = 20
NPRATIO = 4
nrms_loader = "NRMSDataLoaderPretransform"  # NRMSDataLoader

# EXPERIMENT NAME
experiment_name = f"hypertuning-{MODEL_NAME}-debug_{DEBUG}"

#
TRAIN_FRACTION = 1.0 if not DEBUG else 0.0001
TEST_FRACTION = 1.0 if not DEBUG else 0.0001

NRMSLoader_training = (
    NRMSDataLoaderPretransform
    if nrms_loader == "NRMSDataLoaderPretransform"
    else NRMSDataLoader
)
#

optuna_plt_path = DUMP_DIR.joinpath("optuna_plots", f"{experiment_name}-{DT_NOW}")
optuna_plt_path.mkdir(exist_ok=True, parents=True)
mlflow.set_experiment(experiment_name)

#
COLUMNS = [
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
]

##############################
# DATASET
##############################
document_embeddings = "Ekstra_Bladet_word2vec/document_vector.parquet"
DOC_VEC_PATH = PATH.joinpath(f"artifacts/{document_embeddings}")
print("Initiating articles...")
df_articles = pl.read_parquet(DOC_VEC_PATH)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
)
hparams.title_size = 300

df = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE, padding=0
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

df_test = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE, padding=0
    )
    .sample(fraction=TEST_FRACTION, shuffle=True, seed=SEED)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
)

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

test_dataloader = NRMSDataLoader(
    behaviors=df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BS_TEST,
)

test_sample_len = (
    df_test.select(pl.col(DEFAULT_LABELS_COL).list.len()).to_series().to_list()
)
labels_split = df_test.select(pl.col(DEFAULT_LABELS_COL)).to_series().to_list()

# We keep the last day of our training data as the validation set.
last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)


def objective(trial):
    ##############################
    # Hyperparameters
    ##############################
    # Loguniform parameter
    # hparams.learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hparams.learning_rate = trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5])
    # Float parameter
    hparams.dropout = trial.suggest_float("dropout", 0, 0.3, step=0.1)

    hparams.newsencoder_l2_regularization = trial.suggest_categorical(
        "weight_decay", [0, 1e-3, 1e-4, 1e-5]
    )
    # Int parameter
    hparams.history_size = trial.suggest_categorical("history_size", [HISTORY_SIZE])
    # self.head_dim * num_heads == self.embed_dim "embed_dim must be divisible by num_heads" (news_encoder dim 768 for constrastive)
    hparams.head_dim = trial.suggest_categorical("n_head", [16, 20, 24])
    hparams.head_num = trial.suggest_categorical("n_head", [16, 20, 24])
    hparams.attention_hidden_dim = trial.suggest_categorical("d_hid", [100, 200, 300])
    newsencoder_units_per_layer = trial.suggest_categorical(
        "units_per_layer", [32, 64, 128]
    )
    hparams.newsencoder_units_per_layer = [newsencoder_units_per_layer] * 3

    ##############################
    # Training loop (simplified)
    timestamp = dt.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    mlflow_run_name = timestamp  # experiment_name/mlflow_run_name
    tboard_log_dir = DUMP_DIR.joinpath(f"runs/{experiment_name}/{timestamp}")
    MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_OUTPUT_NAME}/weights")

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        mlflow.log_params(
            {
                **{key: val for key, val in trial.params.items()},
                "model": MODEL_NAME,
                "DT": DT_NOW,
                "document_embeddings": document_embeddings,
            }
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tboard_log_dir,
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
        callbacks = [
            tensorboard_callback,
            early_stopping,
            modelcheckpoint,
            lr_scheduler,
        ]
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
        scores = model.scorer.predict(test_dataloader)
        #
        scores_split = tf.split(scores, test_sample_len)
        grouped_aucs = []
        for s, l in tqdm(zip(scores_split, labels_split)):
            try:
                grouped_aucs.append(roc_auc_score(y_true=l, y_score=s))
            except:
                breakpoint()
        grouped_auc_mean = np.mean(grouped_aucs)

        print(f"grouped_auc (val): {grouped_auc_mean:.4f}")
        mlflow.log_metrics({"grouped_auc": grouped_auc_mean})
    return grouped_auc_mean


# Create a study object and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=None)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print(study.trials_dataframe().sort_values("value", ascending=False))
print(study.best_trial.params)

optuna.visualization.matplotlib.plot_optimization_history(study)
plt.savefig(optuna_plt_path.joinpath("plot_optimization_history"), bbox_inches="tight")

optuna.visualization.matplotlib.plot_param_importances(study)
plt.savefig(optuna_plt_path.joinpath("plot_param_importances"), bbox_inches="tight")

# optuna.visualization.matplotlib.plot_parallel_coordinate(study)(study)
# plt.savefig(optuna_plt_path.joinpath("plot_parallel_coordinate"), bbox_inches="tight")

# optuna.visualization.matplotlib.plot_slice(study)(study)
# plt.savefig(optuna_plt_path.joinpath("plot_slice"), bbox_inches="tight")

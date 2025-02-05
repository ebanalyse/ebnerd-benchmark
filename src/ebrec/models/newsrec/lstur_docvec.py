# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, ComputeMasking, OverwriteMasking
from ebrec.models.newsrec.base_model import BaseModel
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

__all__ = ["LSTURModel"]


class LSTURDocVec(BaseModel):
    """Modified LSTUR model(Neural News Recommendation with Multi-Head Self-Attention)
    - Initiated with article-embeddings.

    Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu and Xing Xie:
    Neural News Recommendation with Long- and Short-term User Representations, ACL 2019
    """

    def __init__(
        self,
        hparams,
        seed=None,
    ):
        self.hparams = hparams
        self.seed = seed

        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # BUILD AND COMPILE MODEL:
        self.model, self.scorer = self._build_graph()
        data_loss = self._get_loss(self.hparams.loss)
        train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )
        self.model.compile(loss=data_loss, optimizer=train_optimizer)

    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """

        if optimizer == "adam":
            train_opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")

        return train_opt

    def _build_graph(self):
        """Build LSTUR model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_lstur()
        return model, scorer

    def _build_userencoder(self, titleencoder, type="ini"):
        """The main function to create user encoder of LSTUR.

        Args:
            titleencoder (object): the news encoder of LSTUR.

        Return:
            object: the user encoder of LSTUR.
        """

        his_input_title = keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="float32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        user_embedding_layer = layers.Embedding(
            input_dim=self.hparams.n_users + 1,
            output_dim=self.hparams.gru_unit,  # Dimension of the dense embedding.
            trainable=True,
            embeddings_initializer="zeros",
        )

        long_u_emb = layers.Reshape((self.hparams.gru_unit,))(
            user_embedding_layer(user_indexes)
        )
        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)

        if type == "ini":
            user_present = layers.GRU(
                self.hparams.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(
                layers.Masking(mask_value=0.0)(click_title_presents),
                initial_state=[long_u_emb],
            )
        elif type == "con":
            short_uemb = layers.GRU(
                self.hparams.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(layers.Masking(mask_value=0.0)(click_title_presents))

            user_present = layers.Concatenate()([short_uemb, long_u_emb])
            user_present = layers.Dense(
                self.hparams.gru_unit,
                bias_initializer=keras.initializers.Zeros(),
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            )(user_present)

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, units_per_layer: list[int] = list[64, 64, 64]):
        """The main function to create news encoder of LSTUR.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of LSTUR.
        """
        # LSTUR
        sequences_input_title = keras.Input(
            shape=(self.hparams.title_size), dtype="float32"
        )
        x = sequences_input_title

        # Create configurable Dense layers:
        for layer in units_per_layer:
            x = tf.keras.layers.Dense(
                units=layer,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(
                    self.hparams.newsencoder_l2_regularization
                ),
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.hparams.dropout)(x)

        # # Classic LSTUR:
        # x = layers.Conv1D(
        #     self.hparams.filter_num,
        #     self.hparams.window_size,
        #     activation=self.hparams.cnn_activation,
        #     padding="same",
        #     bias_initializer=keras.initializers.Zeros(),
        #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        # )(x)
        # x = layers.Dropout(self.hparams.dropout)(x)
        # x = layers.Masking()(
        #     OverwriteMasking()([x, ComputeMasking()(sequences_input_title)])
        # )
        # pred_title = AttLayer2(self.hparams.gru_unit, seed=self.seed)(x)

        pred_title = tf.keras.layers.Dense(
            units=self.hparams.gru_unit, activation="relu"
        )(x)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_lstur(self):
        """The main function to create LSTUR's logic. The core of LSTUR
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        his_input_title = keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="float32"
        )
        pred_input_title = keras.Input(
            shape=(None, self.hparams.title_size),
            dtype="float32",
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="float32",
        )
        pred_title_reshape = layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )
        user_indexes = keras.Input(shape=(1,), dtype="float32")

        titleencoder = self._build_newsencoder(
            units_per_layer=self.hparams.newsencoder_units_per_layer
        )
        self.userencoder = self._build_userencoder(titleencoder, type=self.hparams.type)
        self.newsencoder = titleencoder

        user_present = self.userencoder([his_input_title, user_indexes])
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([user_indexes, his_input_title, pred_input_title], preds)
        scorer = keras.Model(
            [user_indexes, his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer

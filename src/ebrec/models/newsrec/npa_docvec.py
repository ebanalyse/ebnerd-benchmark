# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from ebrec.models.newsrec.layers import PersonalizedAttentivePooling
from ebrec.models.newsrec.base_model import BaseModel


__all__ = ["NPAModel"]


class NPADocVec:
    """NPA model(Neural News Recommendation with Attentive Multi-View Learning)

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self,
        hparams,
        seed=None,
    ):
        """Initialization steps for LSTUR."""
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

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_graph(self):
        """Build NPA model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_npa()
        return model, scorer

    def _build_userencoder(self, titleencoder, user_embedding_layer):
        """The main function to create user encoder of NPA.

        Args:
            titleencoder (object): the news encoder of NPA.

        Return:
            object: the user encoder of NPA.
        """

        his_input_title = keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="float32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="float32")

        nuser_id = layers.Reshape((1, 1))(user_indexes)
        repeat_uids = layers.Concatenate(axis=-2)(
            [nuser_id] * self.hparams.history_size
        )
        his_title_uid = layers.Concatenate(axis=-1)([his_input_title, repeat_uids])

        click_title_presents = layers.TimeDistributed(titleencoder)(his_title_uid)

        u_emb = layers.Reshape((self.hparams.user_emb_dim,))(
            user_embedding_layer(user_indexes)
        )
        user_present = PersonalizedAttentivePooling(
            self.hparams.history_size,
            self.hparams.filter_num,
            self.hparams.attention_hidden_dim,
            seed=self.seed,
        )(
            [
                click_title_presents,
                layers.Dense(self.hparams.attention_hidden_dim)(u_emb),
            ]
        )

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(
        self, user_embedding_layer, units_per_layer: list[int] = list[64, 64, 64]
    ):
        """The main function to create news encoder of NPA."""
        sequences_input_title = keras.Input(
            shape=(self.hparams.title_size), dtype="float32"
        )
        sequence_title_uindex = keras.Input(
            shape=(self.hparams.title_size + 1,), dtype="float32"
        )
        # UserEmb:
        sequences_input_title = layers.Lambda(
            lambda x: x[:, : self.hparams.title_size]
        )(sequence_title_uindex)
        user_index = layers.Lambda(lambda x: x[:, self.hparams.title_size :])(
            sequence_title_uindex
        )
        u_emb = layers.Reshape((self.hparams.user_emb_dim,))(
            user_embedding_layer(user_index)
        )
        # Document-vector Encoding:
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

        # y = layers.Dropout(self.hparams.dropout)(x)
        # y = layers.Conv1D(
        #     self.hparams.filter_num,
        #     self.hparams.window_size,
        #     activation=self.hparams.cnn_activation,
        #     padding="same",
        #     bias_initializer=keras.initializers.Zeros(),
        #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        # )(y)
        # y = layers.Dropout(self.hparams.dropout)(y)
        #
        # pred_title = PersonalizedAttentivePooling(
        #     self.hparams.title_size,
        #     self.hparams.filter_num,
        #     self.hparams.attention_hidden_dim,
        #     seed=self.seed,
        # )([x, layers.Dense(self.hparams.attention_hidden_dim)(u_emb)])

        pred_title = layers.Concatenate()(
            [x, layers.Dense(self.hparams.attention_hidden_dim)(u_emb)]
        )
        pred_title = layers.Dense(units=self.hparams.user_emb_dim, activation="relu")(
            pred_title
        )
        model = keras.Model(sequence_title_uindex, pred_title, name="news_encoder")
        return model

    def _build_npa(self):
        """The main function to create NPA's logic. The core of NPA
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and predict.
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
        pred_title_one_reshape = layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )

        user_indexes = keras.Input(shape=(1,), dtype="float32")

        nuser_index = layers.Reshape((1, 1))(user_indexes)

        # Calculate npratio + 1 based on the dynamic shape of pred_input_title
        npratio_plus_one = tf.shape(pred_input_title)[1]

        repeat_uindex = tf.tile(nuser_index, [1, npratio_plus_one, 1])

        pred_title_uindex = layers.Concatenate(axis=-1)(
            [pred_input_title, repeat_uindex]
        )
        pred_title_uindex_one = layers.Concatenate()(
            [pred_title_one_reshape, user_indexes]
        )

        user_embedding_layer = layers.Embedding(
            input_dim=self.hparams.n_users + 1,
            output_dim=self.hparams.user_emb_dim,
            trainable=True,
            embeddings_initializer="zeros",
        )

        titleencoder = self._build_newsencoder(
            units_per_layer=self.hparams.newsencoder_units_per_layer,
            user_embedding_layer=user_embedding_layer,
        )
        userencoder = self._build_userencoder(titleencoder, user_embedding_layer)
        newsencoder = titleencoder

        user_present = userencoder([his_input_title, user_indexes])

        news_present = layers.TimeDistributed(newsencoder)(pred_title_uindex)
        news_present_one = newsencoder(pred_title_uindex_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([user_indexes, his_input_title, pred_input_title], preds)
        scorer = keras.Model(
            [user_indexes, his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer

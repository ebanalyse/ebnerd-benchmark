# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np


class NRMSDocVec:
    """
    Modified NRMS model (Neural News Recommendation with Multi-Head Self-Attention)
    - Initiated with article-embeddings.

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
    """

    def __init__(
        self,
        hparams: dict,
        seed: int = None,
    ):
        """Initialization steps for NRMS."""
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
            train_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")
        return train_opt

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="float32"
        )

        click_title_presents = tf.keras.layers.TimeDistributed(titleencoder)(
            his_input_title
        )
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        model = tf.keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, units_per_layer: list[int] = list[512, 512, 512]):
        """THIS IS OUR IMPLEMENTATION.
        The main function to create a news encoder.

        Parameters:
            units_per_layer (int): The number of neurons in each Dense layer.

        Return:
            object: the news encoder.
        """
        DOCUMENT_VECTOR_DIM = self.hparams.title_size
        OUTPUT_DIM = self.hparams.head_num * self.hparams.head_dim

        # DENSE LAYERS (FINE-TUNED):
        sequences_input_title = tf.keras.Input(
            shape=(DOCUMENT_VECTOR_DIM), dtype="float32"
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

        # OUTPUT:
        pred_title = tf.keras.layers.Dense(units=OUTPUT_DIM, activation="relu")(x)

        # Construct the final model
        model = tf.keras.Model(
            inputs=sequences_input_title, outputs=pred_title, name="news_encoder"
        )

        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size),
            dtype="float32",
        )
        pred_input_title = tf.keras.Input(
            # shape = (hparams.npratio + 1, hparams.title_size)
            shape=(None, self.hparams.title_size),
            dtype="float32",
        )
        pred_input_title_one = tf.keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="float32",
        )
        pred_title_one_reshape = tf.keras.layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )
        titleencoder = self._build_newsencoder(
            units_per_layer=self.hparams.newsencoder_units_per_layer
        )
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_input_title)
        news_present = tf.keras.layers.TimeDistributed(self.newsencoder)(
            pred_input_title
        )
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = tf.keras.layers.Dot(axes=-1)([news_present, user_present])
        preds = tf.keras.layers.Activation(activation="softmax")(preds)

        pred_one = tf.keras.layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)

        model = tf.keras.Model([his_input_title, pred_input_title], preds)
        scorer = tf.keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer

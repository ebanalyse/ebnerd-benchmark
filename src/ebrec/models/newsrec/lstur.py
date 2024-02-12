# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, ComputeMasking, OverwriteMasking
from ebrec.models.newsrec.base_model import BaseModel
from tensorflow.keras import layers
import tensorflow.keras as keras


__all__ = ["LSTURModel"]


class LSTURModel(BaseModel):
    """LSTUR model(Neural News Recommendation with Multi-Head Self-Attention)

    Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu and Xing Xie:
    Neural News Recommendation with Long- and Short-term User Representations, ACL 2019

    Attributes:0
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self,
        hparams,
        word2vec_embedding=None,
        seed=None,
        **kwargs,
    ):
        """Initialization steps for LSTUR.
        Compared with the BaseModel, LSTUR need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as type and gru_unit are there.
        """

        super().__init__(
            hparams=hparams,
            word2vec_embedding=word2vec_embedding,
            seed=seed,
            **kwargs,
        )

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
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
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

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of LSTUR.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of LSTUR.
        """

        sequences_input_title = keras.Input(
            shape=(self.hparams.title_size,), dtype="int32"
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(self.hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            self.hparams.filter_num,
            self.hparams.window_size,
            activation=self.hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(self.hparams.dropout)(y)
        y = layers.Masking()(
            OverwriteMasking()([y, ComputeMasking()(sequences_input_title)])
        )
        pred_title = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)
        print(pred_title)
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
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            # shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
            shape=(None, self.hparams.title_size),
            dtype="int32",
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_reshape = layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word2vec_embedding.shape[1],
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
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

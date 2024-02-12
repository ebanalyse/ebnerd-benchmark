from typing import Any, Dict
from tensorflow import keras
import tensorflow as tf
import numpy as np
import abc

__all__ = ["BaseModel"]


class BaseModel:
    """Basic class of models

    Attributes:
        hparams (object): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
        graph (object): An optional graph.
        seed (int): Random seed.
    """

    def __init__(
        self,
        hparams: Dict[str, Any],
        word2vec_embedding: np.ndarray = None,
        # if 'word2vec_embedding' not provided:
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed=None,
    ):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (object): Hold the entire set of hyperparameters.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # ASSIGN 'hparams':
        self.hparams = hparams

        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = word2vec_embedding

        # BUILD AND COMPILE MODEL:
        self.model, self.scorer = self._build_graph()
        self.loss = self._get_loss(self.hparams.loss)
        self.train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )
        self.model.compile(loss=self.loss, optimizer=self.train_optimizer)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

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
            train_opt = keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")

        return train_opt

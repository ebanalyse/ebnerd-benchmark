#
DEFAULT_TITLE_SIZE = 30
DEFAULT_BODY_SIZE = 40
UNKNOWN_TITLE_VALUE = [0] * DEFAULT_TITLE_SIZE
UNKNOWN_BODY_VALUE = [0] * DEFAULT_BODY_SIZE

DEFAULT_DOCUMENT_SIZE = 768


class hparams_naml:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 50
    body_size: int = DEFAULT_BODY_SIZE
    vert_num: int = 100
    vert_emb_dim: int = 10
    subvert_num: int = 100
    subvert_emb_dim: int = 10
    # MODEL ARCHITECTURE
    dense_activation: str = "relu"
    cnn_activation: str = "relu"
    attention_hidden_dim: int = 200
    filter_num: int = 400
    window_size: int = 3
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 0.0001


class hparams_lstur:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 50
    n_users: int = 50000
    # MODEL ARCHITECTURE
    cnn_activation: str = "relu"
    type: str = "ini"
    attention_hidden_dim: int = 200
    gru_unit: int = 400
    filter_num: int = 400
    window_size: int = 3
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 0.0001


class hparams_npa:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 50
    n_users: int = 50000
    # MODEL ARCHITECTURE
    cnn_activation: str = "relu"
    attention_hidden_dim: int = 200
    user_emb_dim: int = 400
    filter_num: int = 400
    window_size: int = 3
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 0.0001


class hparams_nrms:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 50
    # MODEL ARCHITECTURE
    head_num: int = 20
    head_dim: int = 20
    attention_hidden_dim: int = 200
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 0.0001


class hparams_nrms_docvec:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_DOCUMENT_SIZE
    history_size: int = 50
    # MODEL ARCHITECTURE
    head_num: int = 20
    head_dim: int = 20
    attention_hidden_dim: int = 200
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 0.0001
    newsencoder_units_per_layer: list[int] = [512, 512, 512]

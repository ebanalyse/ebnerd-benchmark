from arguments.args_shared import add_shared_args
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for LSTURModel training"
    )
    # Add shared arguments
    parser = add_shared_args(parser)
    # =====================================================================================
    #  ############################# UNIQUE FOR LSTUR ###############################
    # =====================================================================================
    parser.add_argument(
        "--model",
        type=str,
        default="NPADocVec",
        help="'NPADocVec' and 'LSTURDocVec' the functionality. Hence, this should always be 'LSTURDocVec'",
    )
    parser.add_argument(
        "--document_embeddings",
        type=str,
        default="Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet",
        help="Path to the document embeddings file",
    )
    # Model and loader settings
    parser.add_argument(
        "--title_size",
        type=int,
        default=768,
        help="Maximum length of title encoding",
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=None,
        help="The number of users in the lookup table. If 'None' use all users from the training-set.",
    )
    # MODEL ARCHITECTURE
    parser.add_argument(
        "--attention_hidden_dim",
        type=int,
        default=200,
        help="Attention hidden dim.",
    )
    parser.add_argument(
        "--user_emb_dim",
        type=int,
        default=400,
        help="Dimension of user hidden dim.",
    )
    parser.add_argument(
        "--filter_num",
        type=int,
        default=400,
        help="",
    )
    parser.add_argument(
        "--newsencoder_units_per_layer",
        nargs="+",
        type=int,
        default=[512, 512, 512],
        help="List of units per layer in the news encoder",
    )
    parser.add_argument(
        "--newsencoder_l2_regularization",
        type=float,
        default=1e-4,
        help="L2 regularization for the news encoder",
    )
    # parser.add_argument(
    #     "--window_size",
    #     type=int,
    #     default=3,
    #     help="",
    # )
    # parser.add_argument(
    #     "--cnn_activation",
    #     type=str,
    #     default="relu",
    #     help="Activation function in the CNN.",
    # )
    return parser.parse_args()

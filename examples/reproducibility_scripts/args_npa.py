from args_shared import add_shared_args
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
        default="LSTURModel",
        help="'NPAModel' and 'LSTURModel' the functionality. Hence, this should always be 'LSTURModel'",
    )
    # Transformer settings
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default="FacebookAI/xlm-roberta-large",
        help="Transformer model name",
    )
    # Model and loader settings
    parser.add_argument(
        "--title_size",
        type=int,
        default=30,
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
        "--cnn_activation",
        type=str,
        default="relu",
        help="Activation function in the CNN.",
    )
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
        "--window_size",
        type=int,
        default=3,
        help="",
    )
    return parser.parse_args()

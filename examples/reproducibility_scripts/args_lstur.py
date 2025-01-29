from args_shared import add_shared_args
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for NRMSModel training"
    )
    # Add shared arguments
    parser = add_shared_args(parser)

    # =====================================================================================
    #  ############################# UNIQUE FOR LSTUR ###############################
    # =====================================================================================
    # Model and loader settings

    # Transformer settings
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default="FacebookAI/xlm-roberta-large",
        help="Transformer model name",
    )
    parser.add_argument(
        "--max_title_length",
        type=int,
        default=30,
        help="Maximum length of title encoding",
    )

    # Hyperparameters
    parser.add_argument(
        "--head_num", type=int, default=20, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=20, help="Dimension of each attention head"
    )
    parser.add_argument(
        "--attention_hidden_dim",
        type=int,
        default=200,
        help="Dimension of attention hidden layers",
    )

    # Optimizer settings
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use"
    )
    parser.add_argument(
        "--loss", type=str, default="cross_entropy_loss", help="Loss function"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )

    return parser.parse_args()

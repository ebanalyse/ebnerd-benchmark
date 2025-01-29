from args_shared import add_shared_args
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for NRMSModel training"
    )
    # Add shared arguments
    parser = add_shared_args(parser)
    # =====================================================================================
    #  ############################# UNIQUE FOR NRMSModel ###############################
    # =====================================================================================
    # Model and loader settings
    parser.add_argument(
        "--nrms_loader",
        type=str,
        default="NRMSDataLoaderPretransform",
        choices=["NRMSDataLoaderPretransform", "NRMSDataLoader"],
        help="Data loader type (speed or memory efficient)",
    )
    # Transformer settings
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default="FacebookAI/xlm-roberta-large",
        help="Transformer model name",
    )
    parser.add_argument(
        "--title_size",
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
    return parser.parse_args()

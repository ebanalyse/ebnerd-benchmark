import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for NRMSModel")
    # General settings
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    # Model-specific settings
    parser.add_argument(
        "--model_func", type=str, default="NRMSModel", help="Model function in use"
    )
    parser.add_argument(
        "--datasplit", type=str, default="ebnerd_small", help="Dataset split to use"
    )
    parser.add_argument(
        "--bs_train", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--bs_test", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--batch_size_test_wo_b",
        type=int,
        default=32,
        help="Batch size for testing without balancing",
    )
    parser.add_argument(
        "--batch_size_test_w_b",
        type=int,
        default=4,
        help="Batch size for testing with balancing",
    )
    parser.add_argument(
        "--history_size", type=int, default=20, help="History size for the model"
    )
    parser.add_argument(
        "--npratio", type=int, default=4, help="Negative-positive ratio"
    )

    # Training settings
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use",
    )
    parser.add_argument(
        "--fraction_test",
        type=float,
        default=1.0,
        help="Fraction of testing data to use",
    )

    # Model hyperparameters
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default="Maltehb/danish-bert-botxo",
        help="Transformer model name",
    )
    parser.add_argument(
        "--max_title_length", type=int, default=30, help="Maximum title length"
    )
    parser.add_argument(
        "--head_num", type=int, default=20, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=20, help="Dimension of attention heads"
    )
    parser.add_argument(
        "--attention_hidden_dim",
        type=int,
        default=200,
        help="Hidden dimension of attention",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use"
    )
    parser.add_argument(
        "--loss", type=str, default="cross_entropy_loss", help="Loss function"
    )
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout rate")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )

    return parser.parse_args()

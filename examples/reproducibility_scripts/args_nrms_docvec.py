from args_shared import add_shared_args
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for NRMSDocVec training"
    )
    # Add shared arguments
    parser = add_shared_args(parser)

    # =====================================================================================
    #  ############################# UNIQUE FOR NRMSDocVec ###############################
    # =====================================================================================

    # Model and loader settings
    parser.add_argument(
        "--nrms_loader",
        type=str,
        default="NRMSDataLoaderPretransform",
        choices=["NRMSDataLoaderPretransform", "NRMSDataLoader"],
        help="Data loader type (speed or memory efficient)",
    )

    parser.add_argument(
        "--document_embeddings",
        type=str,
        default="Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet",
        help="Path to the document embeddings file",
    )
    # Model function and architecture
    parser.add_argument(
        "--title_size", type=int, default=768, help="Size of title encoding"
    )
    parser.add_argument(
        "--head_num", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=16, help="Dimension of each attention head"
    )
    parser.add_argument(
        "--attention_hidden_dim",
        type=int,
        default=200,
        help="Dimension of attention hidden layers",
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

    return parser.parse_args()

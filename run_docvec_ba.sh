#!/bin/bash

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./venv

# NRMS
python examples/reproducibility_scripts/ebnerd_nrms_ba.py \
    --title_size 300 \
    --history_size 20 \
    --newsencoder_units_per_layer 256 256 256 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet \

python examples/reproducibility_scripts/ebnerd_nrms_ba.py \
    --title_size 300 \
    --history_size 20 \
    --newsencoder_units_per_layer 512 512 512 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet \

python examples/reproducibility_scripts/ebnerd_nrms_ba.py \
    --title_size 300 \
    --history_size 10 \
    --newsencoder_units_per_layer 256 256 256 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet \

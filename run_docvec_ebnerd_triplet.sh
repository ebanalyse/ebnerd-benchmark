#!/bin/bash

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./venv

# ========= USER INFORMATION =========
# User History: 10 / 20
python examples/reproducibility_scripts/ebnerd_nrms_docvec.py \
    --datasplit ebnerd_small \
    --title_size 300 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet \
    --epochs 5 \
    --bs_train 32 \
    --history_size 10 \
    --npratio 4 \
    --head_num 16 \
    --head_dim 16 \
    --attention_hidden_dim 200 \
    --newsencoder_units_per_layer 256 256 256 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --newsencoder_l2_regularization 1e-4

python examples/reproducibility_scripts/ebnerd_nrms_docvec.py \
    --datasplit ebnerd_small \
    --title_size 300 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet \
    --epochs 5 \
    --bs_train 32 \
    --history_size 20 \
    --npratio 4 \
    --head_num 16 \
    --head_dim 16 \
    --attention_hidden_dim 200 \
    --newsencoder_units_per_layer 256 256 256 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --newsencoder_l2_regularization 1e-4

# ========= MODEL ARCHITECTURE =========
# NN-layer: 256 / 512
python examples/reproducibility_scripts/ebnerd_nrms_docvec.py \
    --datasplit ebnerd_small \
    --title_size 300 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet \
    --epochs 5 \
    --bs_train 32 \
    --history_size 20 \
    --npratio 4 \
    --head_num 16 \
    --head_dim 16 \
    --attention_hidden_dim 200 \
    --newsencoder_units_per_layer 512 512 512 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --newsencoder_l2_regularization 1e-4


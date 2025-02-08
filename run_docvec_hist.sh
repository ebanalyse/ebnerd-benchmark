#!/bin/bash

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./venv

# NRMS
python examples/reproducibility_scripts/ebnerd_nrms_docvec_hist.py \
    --datasplit ebnerd_small \
    --epochs 5 \
    --bs_train 32 \
    --history_size 20 \
    --npratio 4 \
    --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet \
    --head_num 16 \
    --head_dim 16 \
    --attention_hidden_dim 200 \
    --newsencoder_units_per_layer 256 256 256 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --newsencoder_l2_regularization 1e-4

# LSTUR
python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_hist.py \
    --model LSTURDocVec \
    --n_users 0 \
    --datasplit ebnerd_small \
    --epochs 5 \
    --bs_train 32 \
    --history_size 20 \
    --npratio 4 \
    --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet \
    --type ini \
    --gru_unit 400 \
    --newsencoder_units_per_layer 256 256 256 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --newsencoder_l2_regularization 1e-4

# NPA
python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_hist.py \
    --model NPADocVec \
    --n_users 0 \
    --datasplit ebnerd_small \
    --epochs 5 \
    --bs_train 32 \
    --history_size 20 \
    --npratio 4 \
    --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet \
    --attention_hidden_dim 200 \
    --filter_num 400 \
    --newsencoder_units_per_layer 256 256 256 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --newsencoder_l2_regularization 1e-4

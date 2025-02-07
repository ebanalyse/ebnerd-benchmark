#!/bin/bash

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./venv

# NRMS
python examples/reproducibility_scripts/ebnerd_nrms_docvec_emb.py \
    --title_size 300 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet

python examples/reproducibility_scripts/ebnerd_nrms_docvec_emb.py \
    --title_size 768 \
    --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet

python examples/reproducibility_scripts/ebnerd_nrms_docvec_emb.py \
    --title_size 768 \
    --document_embeddings FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet

python examples/reproducibility_scripts/ebnerd_nrms_docvec_emb.py \
    --title_size 768 \
    --document_embeddings google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet

# LSTUR
python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model LSTURDocVec \
    --title_size 300 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet

python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model LSTURDocVec \
    --title_size 768 \
    --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet

python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model LSTURDocVec \
    --title_size 768 \
    --document_embeddings google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet

python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model LSTURDocVec \
    --title_size 768 \
    --document_embeddings FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet

# NPA
python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model NPADocVec \
    --title_size 300 \
    --document_embeddings Ekstra_Bladet_word2vec/document_vector.parquet

python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model NPADocVec \
    --title_size 768 \
    --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet

python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model NPADocVec \
    --title_size 768 \
    --document_embeddings google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet

python examples/reproducibility_scripts/ebnerd_lstur_npa_docvec_emb.py \
    --model NPADocVec \
    --title_size 768 \
    --document_embeddings FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet


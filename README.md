# Contributors
<p align="left">
  <img src="https://contributors-img.web.app/image?repo=ebanalyse/ebnerd-benchmark" width = 50/>
</p>


# Ekstra Bladet News Recommendation Dataset (EB-NeRD)

This repository serves as a **toolbox** for working with the **Ekstra Bladet News Recommendation Dataset (EB-NeRD)**—a rich dataset designed to advance research and benchmarking in news recommendation systems.  

EB-NeRD is based on user behavior logs from **[Ekstra Bladet](https://ekstrabladet.dk/)**, a classical Danish newspaper published by **[JP/Politikens Media Group](https://jppol.dk/en/)** in Copenhagen. The dataset was created as part of the **[18th ACM Conference on Recommender Systems Challenge](https://recsys.acm.org/recsys24/challenge/)** (**RecSys'24 Challenge**).

## What You'll Find Here
This repository provides:
- **Starter notebooks** for descriptive data analysis, data preprocessing, and baseline modeling.
- **Examples of established models** to kickstart experimentation.
<!-- - **A step-by-step tutorial** for running a **CodaBench server locally**, which is required to evaluate models on the hidden test set. -->

## Useful Links
For more information about the dataset, the RecSys '24 Challenge, and its usage, please visit: **[recsys.eb.dk](https://recsys.eb.dk/)**.

<!-- ### CodaBench
- **[CodaBench Server Setup Guide](LINK)** -->

### Papers
- **[RecSys Challenge Paper](https://dl.acm.org/doi/10.1145/3640457.3687164)**
- **[Dataset Paper](https://dl.acm.org/doi/10.1145/3687151.3687152)**
- **[RecSys'24 Challenge Proceedings](https://dl.acm.org/doi/proceedings/10.1145/3687151)**

---

# Getting Started


We recommend using [**conda**](https://docs.conda.io/projects/conda/en/latest/glossary.html#conda-environment) for environment.

## Installation

```bash
# 1. Create and activate a new conda environment
conda create -n <environment_name> python=3.11
conda activate <environment_name>

# 2. Clone this repo within VSCode or using command line:
git clone https://github.com/ebanalyse/ebnerd-benchmark.git

# 3. Install the core ebrec package to the enviroment:
pip install .
```

### M1 Mac Users
We have experienced issues installing *tensorflow* for M1 Macbooks (`sys_platform == 'darwin'`) when using conda. To avoid this, we suggest to use venv if running on macbooks.

We have encountered issues installing *TensorFlow* on M1 MacBooks when using conda (i.e., `sys_platform == 'darwin'`).
**Workaround**: Use `venv` instead of `conda`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Alternatively, install ```.venv``` directly in the project folder using conda:
```bash
conda create -p .venv python=3.11.8
conda activate ./.venv
```

### GPU Support
To enable GPU support, install the appropriate TensorFlow package based on your platform:
```bash
# For Linux
pip install tensorflow-gpu
```
```bash
# For macOS
pip install tensorflow-macos
```

---

# Algorithms
To get started quickly, we have implemented several **news recommender systems**, including:

| Model | Notebook | Example |
|-------|----------|---------|
| [**NRMS**](https://aclanthology.org/D19-1671/) | [NRMS Notebook](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/nrms_ebnerd.ipynb) | [NRMS Example](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/nrms_dummy.py) |
| [**LSTUR**](https://aclanthology.org/P19-1033/) | - | [LSTUR Example](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/lstur_dummy.py) |
| [**NPA**](https://arxiv.org/abs/1907.05559) | - | [NPA Example](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/npa_dummy.py) |
| [**NAML**](https://arxiv.org/abs/1907.05576) | - | [NAML Example](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/naml_dummy.py) |
| **NRMSDocVec** | - | [NRMSDocVec Example](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/nrms_docvec_dummy.py) |

The implementations of **NRMS**, **LSTUR**, **NPA**, and **NAML** are adapted from the excellent [**recommenders**](https://github.com/recommenders-team/recommenders) repository, with all non-model-related code removed for simplicity. 
**NRMSDocVec** is our variation of **NRMS** where the *NewsEncoder* is initialized with **document embeddings** (i.e., article embeddings generated from a pretrained language model), rather than learning embeddings solely from scratch.

---

## Data Manipulation & Enrichment

To help you get started, we have created a set of **introductory notebooks** designed for quick experimentation, including:

- [**ebnerd_descriptive_analysis**](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/datasets/ebnerd_descriptive_analysis.ipynb): Basic descriptive analysis of EB-NeRD.
- [**ebnerd_overview**](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/datasets/ebnerd_overview.ipynb): Demonstrates how to join user histories and create binary labels.

*Note: These notebooks were developed on macOS. Small adjustments may be required for other operating systems.*

---

# Reproduce EB-NeRD Experiments

Make sure you’ve installed the repository and dependencies. Then activate your environment:

Activate your enviroment:
```bash
conda activate <environment_name>
```

## [NRMSModel](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/src/ebrec/models/newsrec/nrms.py) 
```bash
python examples/reproducibility_scripts/ebnerd_nrms.py
  --datasplit ebnerd_small \
  --epochs 5 \
  --bs_train 32 \
  --bs_test 32 \
  --history_size 20 \
  --npratio 4 \
  --transformer_model_name FacebookAI/xlm-roberta-large \
  --max_title_length 30 \
  --head_num 20 \
  --head_dim 20 \
  --attention_hidden_dim 200 \
  --learning_rate 1e-4 \
  --dropout 0.20
```

### Tensorboards:
```bash
tensorboard --logdir=ebnerd_predictions/runs
```

### [NRMSDocVec](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/src/ebrec/models/newsrec/nrms_docvec.py) 

```bash
python examples/reproducibility_scripts/ebnerd_nrms_docvec.py \
  --datasplit ebnerd_small \
  --epochs 5 \
  --bs_train 32 \
  --history_size 20 \
  --npratio 4 \
  --document_embeddings Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet \
  --head_num 16 \
  --head_dim 16 \
  --attention_hidden_dim 200 \
  --newsencoder_units_per_layer 512 512 512 \
  --learning_rate 1e-4 \
  --dropout 0.2 \
  --newsencoder_l2_regularization 1e-4
```

### Tensorboards:
```bash
tensorboard --logdir=ebnerd_predictions/runs
```
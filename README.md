# Ekstra Bladet News Recommendation Dataset (EB-NeRD)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/github/license/ebanalyse/ebnerd-benchmark" alt="License">
  <img src="https://img.shields.io/badge/framework-TensorFlow%20%7C%20PyTorch-orange" alt="Frameworks">
  <img src="https://img.shields.io/github/stars/ebanalyse/ebnerd-benchmark?style=social" alt="GitHub Stars">
</p>

## Contributor 
<p align="center">
<img src="https://contributors-img.web.app/image?repo=ebanalyse/ebnerd-benchmark" width="50" alt="Contributors"/>
</p>

## About

This repository serves as a **toolbox** for working with the **Ekstra Bladet News Recommendation Dataset (EB-NeRD)**—a rich dataset designed to advance research and benchmarking in news recommendation systems.  

EB-NeRD is based on user behavior logs from **[Ekstra Bladet](https://ekstrabladet.dk/)**, a classical Danish newspaper published by **[JP/Politikens Media Group](https://jppol.dk/en/)** in Copenhagen. The dataset was created as part of the **[18th ACM Conference on Recommender Systems Challenge](https://recsys.acm.org/recsys24/challenge/)** (**RecSys'24 Challenge**).

## What You'll Find Here
This repository provides:
- **Starter notebooks** for descriptive data analysis, data preprocessing, and baseline modeling.
- **Examples of established models** to kickstart experimentation.
- **A step-by-step tutorial** for running a **CodaBench server locally**, which is required to evaluate models on the hidden test set.

## 🔗 Useful Links

| Resource | Description |
|----------|-------------|
| **[recsys.eb.dk](https://recsys.eb.dk/)** | Main dataset website with detailed documentation |
| **[CodaBench Setup Guide](https://github.com/ebanalyse/ebnerd-benchmark/tree/main/codabench)** | Local evaluation server setup instructions |
| **[RecSys'24 Challenge](https://recsys.acm.org/recsys24/challenge/)** | Original competition page |

## ℹ️ Dataset Information

- **📊 Size**: Large-scale news recommendation dataset
- **🏢 Source**: Ekstra Bladet (Danish newspaper)
- **📅 Period**: User behavior logs from JP/Politikens Media Group
- **🎯 Focus**: Balancing accuracy and editorial values in news recommendations

## 📚 Key Papers / References

Below are important papers related to this repository:

| Title | Authors | Venue / Year  |
|-------|---------|--------------|
| [*RecSys Challenge 2024: Balancing Accuracy and Editorial Values in News Recommendation*](https://dl.acm.org/doi/10.1145/3640457.3687164) | J. Kruse, K. Lindskow, S. Kalloori, M. Polignano, C. Pomo, A. Srivastava, A. Uppal, M. R. Andersen, J. Frellsen | Proc. ACM RecSys ’24 |
| [*EB-NeRD: A large-scale dataset for news recommendation*](https://dl.acm.org/doi/10.1145/3687151.3687152) | J. Kruse, K. Lindskow, S. Kalloori, M. Polignano, C. Pomo, A. Srivastava, A. Uppal, M. R. Andersen, J. Frellsen | Proc. RecSys Challenge ’24 |
| [*Proceedings of the RecSys ’24: 18th ACM Conference on Recommender Systems*](https://dl.acm.org/doi/proceedings/10.1145/3687151) | — (conference proceedings volume) | Proc. RecSys ’24 |
| [*Why design choices matter in recommender systems*](https://rdcu.be/esXqW) | J. Kruse, K. Lindskow, M. R. Andersen, J. Frellsen| Nature Machine Intelligence vol. 7 (2025) |

---

# 🚀 Quick Start

Want to jump right in? Here's a 5-minute setup:

```bash
# 1. Clone and install
git clone https://github.com/ebanalyse/ebnerd-benchmark.git
cd ebnerd-benchmark
pip install .

# 2. Run your first model
python examples/quick_start/nrms_dummy.py

# 3. Explore the data (optional)
# Open examples/datasets/ebnerd_overview.ipynb in Jupyter
```

---

# 🛠️ Installation

We recommend using [**conda**](https://docs.conda.io/projects/conda/en/latest/glossary.html#conda-environment) for environment management.

## Prerequisites

- **Python**: 3.10+ (recommended: 3.11)
- **RAM**: Minimum 8GB (16GB+ recommended for larger datasets)
- **Storage**: ~2GB for repository + dataset storage space
- **OS**: Linux, macOS, or Windows

## Standard Installation

```bash
# 1. Create and activate a new conda environment
conda create -n <environment_name> python=3.11
conda activate <environment_name>

# 2. Clone this repo within VSCode or using command line:
git clone https://github.com/ebanalyse/ebnerd-benchmark.git
cd ebnerd-benchmark

# 3. Install the core ebrec package to the environment:
pip install .

# 4. Verify installation
python -c "import ebrec; print('✅ Installation successful!')"
```

### 🍎 M1 Mac Users

We have encountered issues installing *TensorFlow* on M1 MacBooks when using conda (`sys_platform == 'darwin'`).

**Recommended Workaround - Use `venv`:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

**Alternative - Conda with local environment:**
```bash
conda create -p .venv python=3.11.8
conda activate ./.venv
pip install .
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

# 🔬 Algorithms

We have implemented several state-of-the-art **news recommender systems** to get you started quickly:

| Model | Description | Notebook | Example |
|-------|-------------|----------|---------|
| [**NRMS**](https://aclanthology.org/D19-1671/) | Neural News Recommendation with Multi-Head Self-Attention | [📓 Notebook](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/nrms_ebnerd.ipynb) | [🔗 Code](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/nrms_dummy.py) |
| [**LSTUR**](https://aclanthology.org/P19-1033/) | Long- and Short-term User Representations for news recommendation | - | [🔗 Code](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/lstur_dummy.py) |
| [**NPA**](https://arxiv.org/abs/1907.05559) | Neural News Recommendation with Personalized Attention | - | [🔗 Code](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/npa_dummy.py) |
| [**NAML**](https://arxiv.org/abs/1907.05576) | Neural News Recommendation with Attentive Multi-View Learning | - | [🔗 Code](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/naml_dummy.py) |
| **NRMSDocVec** | NRMS variant using pre-trained document embeddings | - | [🔗 Code](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/nrms_docvec_dummy.py) |

The implementations of **NRMS**, **LSTUR**, **NPA**, and **NAML** are adapted from the excellent [**recommenders**](https://github.com/recommenders-team/recommenders) repository, with all non-model-related code removed for simplicity. 
**NRMSDocVec** is our variation of **NRMS** where the *NewsEncoder* is initialized with **document embeddings** (i.e., article embeddings generated from a pretrained language model), rather than learning embeddings solely from scratch.

---

## 📊 Data Manipulation & Enrichment

To help you get started, we have created a set of **introductory notebooks** designed for quick experimentation, including:

- [**ebnerd_descriptive_analysis**](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/datasets/ebnerd_descriptive_analysis.ipynb): Basic descriptive analysis of EB-NeRD.
- [**ebnerd_overview**](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/datasets/ebnerd_overview.ipynb): Demonstrates how to join user histories and create binary labels.

*Note: These notebooks were developed on macOS. Small adjustments may be required for other operating systems.*

---

# 🔄 Reproduce EB-NeRD Experiments

Make sure you've installed the repository and dependencies. Then activate your environment:

Activate your environment:
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

---

## 🛠️ Troubleshooting

### Common Issues

**ImportError: No module named 'ebrec'**
```bash
# Make sure you're in the right environment and have installed the package
conda activate <environment_name>
pip install -e .
```

**TensorFlow GPU not found**
```bash
# Verify GPU installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Memory errors during training**
- Reduce batch size (`--bs_train`, `--bs_test`)
- Reduce history size (`--history_size`)
- Use gradient checkpointing

**M1 Mac TensorFlow issues**
- Use the venv workaround mentioned in installation
- Consider using `tensorflow-macos` instead

### Getting Help

- 📫 **Issues**: [GitHub Issues](https://github.com/ebanalyse/ebnerd-benchmark/issues)

---

# 📄 Cite Our Work

If you use this repository, our methods, or datasets in your research, please cite the following papers:

```bibtex
@inproceedings{kruse2024recsys_challenge,
  author    = {Kruse, Johannes and Lindskow, Kasper and Kalloori, Saikishore and Polignano, Marco and Pomo, Claudio and Srivastava, Abhishek and Uppal, Anshuk and Andersen, Michael Riis and Frellsen, Jes},
  title     = {RecSys Challenge 2024: Balancing Accuracy and Editorial Values in News Recommendations},
  booktitle = {Proceedings of the 18th ACM Conference on Recommender Systems},
  series    = {RecSys '24},
  year      = {2024},
  pages     = {1195--1199},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3640457.3687164},
  url       = {https://doi.org/10.1145/3640457.3687164},
  keywords  = {Beyond-Accuracy, Competition, Dataset, Editorial Values, News Recommendations, Recommender Systems}
}
```

```bibtex
@inproceedings{kruse2024ebnerd,
  author    = {Kruse, Johannes and Lindskow, Kasper and Kalloori, Saikishore and Polignano, Marco and Pomo, Claudio and Srivastava, Abhishek and Uppal, Anshuk and Andersen, Michael Riis and Frellsen, Jes},
  title     = {EB-NeRD: A Large-scale Dataset for News Recommendation},
  booktitle = {Proceedings of the Recommender Systems Challenge 2024},
  series    = {RecSysChallenge '24},
  year      = {2024},
  pages     = {1--11},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3687151.3687152},
  url       = {https://doi.org/10.1145/3687151.3687152},
  keywords  = {Beyond-Accuracy, Dataset, Editorial Values, News Recommendations, Recommender Systems}
}
```

```bibtex
@article{kruse2025design_choices,
  author    = {Kruse, Johannes and Lindskow, Kasper and Andersen, Michael Riis and Frellsen, Jes},
  title     = {Why Design Choices Matter in Recommender Systems},
  journal   = {Nature Machine Intelligence},
  year      = {2025},
  volume    = {7},
  number    = {6},
  pages     = {979--980},
  doi       = {10.1038/s42256-025-01043-5},
  url       = {https://doi.org/10.1038/s42256-025-01043-5},
  publisher = {Nature},
  note      = {In press}
}
```
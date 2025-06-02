# CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-orange?logo=HuggingFace&style=flat-square)](https://huggingface.co/datasets/Exploration-Lab/CoMuMDR)
[![Arxiv](https://img.shields.io/badge/Arxiv-Paper-blue?logo=arxiv&style=flat-square)]()

## Introduction
We present a novel code-mixed multi-modal multi-domain corpus for discourse parsing, named CoMuMDR. This corpus is designed to facilitate research in discourse parsing across multiple languages, modalities and domains, specifically focusing on code-mixed (Hindi + English = Hinglish) data. This repository contains the code for our paper on **CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations**. The paper presents and compares various baseline modeling approaches for discourse parsing on CoMuMDR and popular discourse datasets, used in prior research.

## Abstract
Discourse parsing is an important task useful for NLU applications such as summarization, machine comprehension, and emotion recognition. The current discourse parsing datasets based on conversations consists of written English dialogues restricted to a single domain. In this resource paper, we introduce CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations. The corpus (code-mixed in Hindi and English) has both audio and transcribed text and is annotated with nine discourse relations. We experiment with various SoTA baseline models; the poor performance of SoTA models highlights the challenges of multi-domain code-mixed corpus, pointing towards the need for developing better models for such realistic settings.

## Dataset

The dataset is available at [Exploration-Lab/CoMuMDR](https://huggingface.co/datasets/Exploration-Lab/CoMuMDR).

## Repository Structure

The repository is organized as follows:

- `hierarchical/` - Implementation of the hierarchical baseline model
- `sadpmd/` - Implementation of the SADPMD baseline model
- `sddp/` - Implementation of the SDDP baseline model
- `struct-aware/` - Implementation of the structure-aware baseline model

## Reproducing Results

To reproduce the results presented in our paper, please refer to the `README.md` files in each of the baseline modeling directories:

- For the hierarchical baseline, see [hierarchical/README.md](hierarchical/README.md)
- For the SADPMD baseline, see [sadpmd/README.md](sadpmd/README.md)
- For the SDDP baseline, see [sddp/README.md](sddp/README.md)
- For the structure-aware baseline, see [struct-aware/README.md](struct-aware/README.md)

Each README provides specific instructions on:
- Setting up the environment
- Preprocessing the data
- Training the models
- Evaluating the models
- Reproducing the scores presented in the paper


## Evaluation

The evaluation scores from our experiments can be found in `epoch_scores.csv` and visualized in `epoch_scores.pdf`.

## Contact

For any questions or issues, please contact `{divyaksh,ashutoshm}@cse.iitk.ac.in`


# CoMuMDR

This repository contains the code and data for our paper on Code-. The paper presents various baseline modeling approaches for discourse parsing.

## Repository Structure

The repository is organized as follows:

- `comumdr_data/` - Contains the dataset split into train, dev, and test sets
- `hierarchical/` - Implementation of the hierarchical baseline model
- `sadpmd/` - Implementation of the SADPMD baseline model
- `sddp/` - Implementation of the SDDP baseline model
- `struct-aware/` - Implementation of the structure-aware baseline model

## Reproducing Results

To reproduce the results presented in our paper, please refer to the README.md files in each of the baseline modeling directories:

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

## Dataset

The dataset is available in the `comumdr_data/` directory, split into:
- `train.json` - Training data
- `dev.json` - Development/validation data
- `test.json` - Test data

## Evaluation

The evaluation scores from our experiments can be found in `epoch_scores.csv` and visualized in `epoch_scores.pdf`.

## Contact

For any questions or issues, please open an issue in this repository.


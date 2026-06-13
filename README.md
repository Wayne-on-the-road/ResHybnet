# ResHybnet: Graph-Empowered Insider Threat Detection Based on Daily Activities

This repository provides the experimental code, sample data, intermediate results, and demonstration pipeline for the paper:

**A graph empowered insider threat detection framework based on daily activities**
Wei Hong, Jiao Yin, Mingshan You, Hua Wang, Jinli Cao, Jianxin Li, Ming Liu, Chengyuan Man
*ISA Transactions*, Volume 141, pp. 84–92
Elsevier, 2023

## Overview

Insider threat detection is a challenging cybersecurity task because malicious activities are often sparse, behaviourally subtle, and hidden within large-scale daily enterprise activity logs. This project implements **ResHybnet**, a graph-empowered insider threat detection framework designed to model user-day activities through both sequential behavioural representations and graph-based relational information.

The framework is based on daily user activities from the CERT 4.2 insider threat dataset. It uses activity-sequence modelling to extract behavioural features and then performs insider threat detection using the proposed ResHybnet model. The repository also includes sample data and output folders to support demonstration, reproduction, and further experimentation.

## Repository Structure

```text
.
├── code_for_sequence_process/
│   ├── cert4.2_data/
│   ├── sample_data/
│   ├── 1-to_sequence_for_all_user_multiprocessing.py
│   └── 2-selectout_sample_data_for_LSTM_train.py
├── detection_result/
│   ├── ResHybnet_compare_sequence_length_10round/
│   └── readme.txt
├── lstm_feature_extraction_result/
├── sample_data/
├── tool_and_model/
├── detection_with_ResHybnet.py
├── lstm_feature_extraction.py
└── README.md
```

## Dataset

The experiments are based on a sampled graph dataset constructed from the **CERT 4.2 insider threat dataset**. The sampled user-day nodes and train-test split follow the setting used in the authors' previous work, **B-CITD**.

Because the original CERT 4.2 raw log files are too large to host directly in this repository, users who wish to reproduce the full sequence preprocessing stage need to download the following raw files from the CERT 4.2 dataset source:

```text
logon.csv
device.csv
file.csv
email.csv
```

After downloading these files, place them in the following directory:

```text
code_for_sequence_process/cert4.2_data/
```

The repository also includes prepared sample data, so users can directly run the LSTM feature extraction and ResHybnet detection experiments without repeating the full raw-data preprocessing procedure.

## Workflow

The project consists of three main stages.

### 0. Sequence Preprocessing

The sequence preprocessing scripts are located in:

```text
code_for_sequence_process/
```

This stage converts raw CERT 4.2 activity logs into daily user activity sequences. To reproduce this stage, run the following scripts in order:

```bash
python code_for_sequence_process/1-to_sequence_for_all_user_multiprocessing.py
python code_for_sequence_process/2-selectout_sample_data_for_LSTM_train.py
```

The generated user-day activity sequences are used as the input data for the LSTM-based feature extraction stage. This preprocessing directory is mainly provided for demonstration and reproducibility. It is not necessary to execute this stage if you use the prepared `sample_data/` directory included in the main branch.

### 1. LSTM Autoencoder Feature Extraction

The LSTM autoencoder is used to extract sequential behavioural features from daily user activity sequences. To run this stage, make sure that the `sample_data/` directory is available in the main project directory, and then execute:

```bash
python lstm_feature_extraction.py
```

After training, the extracted feature outputs will be saved to:

```text
lstm_feature_extraction_result/
```

The activity sequence length and related model settings can be adjusted inside the script.

### 2. Insider Threat Detection with ResHybnet

The final detection stage uses the ResHybnet model to identify insider threats based on the prepared behavioural and graph-related representations. To run the detection experiment, execute:

```bash
python detection_with_ResHybnet.py
```

After training and evaluation, the output files will be saved to:

```text
detection_result/
```

Different model configurations, component settings, and feature combinations can be adjusted inside the detection script.

## Quick Start

Users who want to run the main experimental pipeline directly may follow the simplified workflow below:

```bash
# Step 1: Extract LSTM-based sequential features
python lstm_feature_extraction.py

# Step 2: Run insider threat detection with ResHybnet
python detection_with_ResHybnet.py
```

The full sequence preprocessing stage is only required when reconstructing the sample sequence data from the original CERT 4.2 raw log files.

## Outputs

The main output directories are:

```text
lstm_feature_extraction_result/
detection_result/
```

The `lstm_feature_extraction_result/` directory stores the sequential feature extraction results generated by the LSTM autoencoder.

The `detection_result/` directory stores the detection results produced by ResHybnet under different model settings, feature combinations, and sequence-length configurations.

## Reproducibility Notes

This repository is intended to support academic reproducibility and provide a transparent implementation reference for the proposed framework. Users should note the following:

* The full CERT 4.2 raw dataset is not included due to its size.

* Prepared sample data are included for running the main feature extraction and detection stages.

* File paths may need to be adjusted depending on the local operating system and directory structure.

* Model hyperparameters, sequence length, feature combinations, and experimental settings can be modified inside the corresponding Python scripts.

* The preprocessing scripts are provided mainly to demonstrate how daily user activity sequences are generated from raw CERT activity logs.

## Citation

If you use this repository, code, data processing pipeline, or experimental results in your research, please cite the following paper:

```bibtex
@article{hong2023graph,
  title     = {A graph empowered insider threat detection framework based on daily activities},
  author    = {Hong, Wei and Yin, Jiao and You, Mingshan and Wang, Hua and Cao, Jinli and Li, Jianxin and Liu, Ming and Man, Chengyuan},
  journal   = {ISA Transactions},
  volume    = {141},
  pages     = {84--92},
  year      = {2023},
  publisher = {Elsevier}
}
```

## Keywords

Insider threat detection; ResHybnet; graph-based detection; daily activity modelling; LSTM autoencoder; behavioural analytics; cybersecurity; CERT 4.2 dataset.

## Contact

For questions about the code, dataset preparation, or experimental settings, please open an issue in this repository.

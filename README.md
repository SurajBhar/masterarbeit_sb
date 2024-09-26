# Master Thesis Code Repository: Improved Driver Distraction Detection Using Self-Supervised Learning

This repository contains the source code and experimental setup for **Suraj Bhardwaj's** master thesis titled **"Improved Driver Distraction Detection Using Self-Supervised Learning"**. 
The project focuses on detecting driver distractions using Vision Transformers trained using supervised and self-supervised learning techniques.
This work also proposes a novel unsupervised learning based sampling technique (Clustered Feature Weighting: CFW) for loading the imbalanced datasets batchwise during model training. 
CFW improves the imbalance in the dataset batcwise and is a label free approach to solve the problem of learning from imbalanced datasets.
This repository provides all the necessary resources for running the experiments and reproducing the results presented in the thesis.

---

## Thesis Information

![Alt text](thesis_title_page.png)

- **Title**: **"Improved Driver Distraction Detection Using Self-Supervised Learning"**
- **Author**: Suraj Bhardwaj
- **Institution**: University of Siegen, Faculty of Electrical Engineering and Computer Science
- **Program**: Master of Science (M.Sc.) in International Graduate Studies in Mechatronics
- **Examiner 1**: Prof. Dr. Michael Möller (Head of Computer Vision Group, University of Siegen)
- **Examiner 2**: Dr. Jovita Lukasik (Post-Doctoral Researcher, Computer Vision Group, University of Siegen)
- **External Supervisor**: David Lerch M.Sc. (Perceptual User Interface Group, Fraunhofer IOSB, Karlsruhe)
- **Date of Submission**: 15 May 2024

---

## Clustered Feature Weighting
![Alt text](CFW_Algorithm.png)

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Experiments Overview](#experiments-overview)
- [Running Experiments](#running-experiments)
- [Data Preprocessing](#data-preprocessing)
- [Reproducibility](#reproducibility)
- [Dependencies](#dependencies)
- [License](#license)

---

## Repository Structure

The repository is organized as follows:

```bash
.
|-- Vit_baseline.egg-info/        # Metadata for the project
|-- notebooks/                    # Jupyter notebooks for data preprocessing
|   |-- eda_splits.ipynb          # Exploratory Data Analysis (EDA) on the dataset splits
|-- src/
|   |-- components/               # Source code for all experiments
|   |   |-- Grid_Search/          # Hyperparameter search code
|   |   |-- Trainer_D_2/          # Code for training experiment D_2
|   |   |-- distraction_detection_d_a/  # Distraction Detection using Dataloader A (Traditional)
|   |   |-- distraction_detection_d_b/  # Distraction Detection using Dataloader B (CFW)
|   |   |-- dinov2_linear/        # Linear evaluation code for DINOv2
|   |   |-- config_manager/       # Configuration management for experiments
|   |   |-- config_manager_baseline.py  # Baseline config manager
|   |   |-- base_d1_test.py       # Base test file for Dataloader D_1
|   |   |-- data_extraction.py    # Data extraction functions
|   |   |-- data_setup.py         # Dataset setup and loading
|   |   |-- dataset.py            # Dataset handling and preprocessing logic
|   |   |-- engine.py             # Training and model management code
|   |   |-- model_vit.py          # Vision Transformer model definition
|   |   |-- single_gpu_trainer_d1.py  # Single GPU trainer for Dataloader D_1
|   |   |-- test_exp.ipynb        # Experiment testing notebook
|   |   |-- trainer_mod_d1.py     # Modified trainer for Dataloader D_1
|   |   |-- utils.py              # Utility functions
|-- class_image_counts_and_ratios_split_0_train.csv   # Class counts and ratios for train data
|-- class_image_counts_and_ratios_split_0_val.csv     # Class counts and ratios for validation data
|-- class_image_counts_and_ratios_split_0_test.csv    # Class counts and ratios for test data
|-- class_ratios_spli_0_train_output.png              # Class ratios visualization (train)
|-- class_ratios_split_0_val_output.png               # Class ratios visualization (validation)
|-- class_ratios_split_0_test_output.png              # Class ratios visualization (test)
|-- Suraj_Bhardwaj_M_Thesis.pdf  # PDF of the full thesis
|-- .gitignore                   # Files to ignore during version control
|-- README.md                    # This file
|-- requirements.txt             # Project dependencies
|-- setup.py                     # Setup script for package installation
```

---

## Installation

To reproduce the experiments or run any part of this codebase, follow the steps below:

### Clone the repository:
```bash
git clone https://github.com/user1168/masterarbeit_sb.git
cd masterarbeit_sb
```
### Install Dependencies:
All required dependencies can be installed using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Alternatively, you can recreate the conda environments used for ViT and DINOv2 using the provided files:

- **For Vision Transformer (ViT):**
  ```bash
  conda create --name vi_trans --file vi_trans_conda_env.txt
  conda activate vi_trans
  ```

- **For DINOv2:**
  ```bash
  conda create --name dinov2 --file dinov2_conda_env.txt
  conda activate dinov2
  ```

---

## Experiments Overview

The experiments in this repository are organized under the `src/components/` directory. Below is a brief overview of each folder:

- **Grid_Search**: Contains the code for hyperparameter optimization using a grid search approach.
  
- **Trainer_D_2**: Implements the training process for Experiment D_2.
  
- **distraction_detection_d_a**: Code for distraction detection using traditional data loading (Dataloader A).

- **distraction_detection_d_b**: Code for distraction detection using Clustered Feature Weighing (Dataloader B).

- **dinov2_linear**: Code for linear evaluation using the DINOv2 self-supervised model.

---

## Running Experiments

### 1. Grid Search for Hyperparameter Optimization
To perform hyperparameter optimization for the Vision Transformer model, use the scripts in `Grid_Search/`. Update the configuration parameters as needed in the `config_manager/`.
An example of how to run the python modules:
```bash
python src/components/Grid_Search/multi_trainer_grid_search.py --config src/components/Grid_Search/grid_configs/text_files_optim/Exp_07_Adam_0_03.txt
```

### 2. Distraction Detection (Dataloader A - Traditional)
To run the distraction detection experiment using the traditional dataloader (Dataloader A), navigate to `distraction_detection_d_a/` and execute:

```bash
# Use this environment for vit base 16 encoder based experiments
conda activate vi_trans
python src/components/distraction_detection_d_a/vit_baseline_rgb_a_no_aug_trainer.py --config path/to/exp_config.txt
```

### 3. Distraction Detection (Dataloader B - Clustered Feature Weighing)
To run the distraction detection experiment using the Clustered Feature Weighing dataloader (Dataloader B), navigate to `distraction_detection_d_b/` and execute:

```bash
# Use this environment for dinov2 based experiments
conda activate dinov2
export LD_LIBRARY_PATH=/usr/lib/xorg-nvidia-525.116.04/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/xorg/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/xorg-nvidia-535.113.01/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python src/components/dinov2_linear/dataloader_b_dino_v2/dinov2_d_b_linear_no_aug.py --config path/to/exp_config.txt
```

---

## Data Preprocessing

All data preprocessing steps, including exploratory data analysis (EDA), are contained in the `notebooks/` directory. You can explore and process the dataset splits by running the notebooks:
An example of extracting the Kinect IR Right top dataset:
```bash
jupyter notebook notebooks/data_extraction_ir/data_extrc_kinect_ir_right_top.ipynb
```

---

## Reproducibility

To ensure reproducibility, the following steps have been taken:

- **Config Management**: Configurations for all experiments are managed using configuration files in `config_manager/`. Modify these to adjust parameters like learning rate, batch size, epochs, and optimizer.
  
- **Environment Setup**: Dependency files for both Vision Transformer and DINOv2 environments are provided. Make sure to install them using the conda environments outlined in the installation section.

- **Balanced Accuracy**: Since the dataset is imbalanced, the evaluation metric used for hyperparameter optimization and experiments is balanced accuracy instead of traditional accuracy.

---

## Dependencies

Key dependencies for this project include:

- Python 3.9
- PyTorch (for Vision Transformer models)
- torchvision
- scikit-learn
- Jupyter Notebook (for data preprocessing)
- Additional dependencies listed in `requirements.txt`.

---

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT) - feel free to modify and distribute the code with proper attribution.

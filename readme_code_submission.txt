Name: Suraj Bhardwaj
Mat. Nr.: 1531066
Master Thesis : Improved Driver Distraction Detection Using Self-Supervised Learning


############# Directories ##################

Data Extraction: notebooks/data_extraction_ir
Data Reorganisation: notebooks/data_reorganisation
Dataloader Experiments:src/components/distraction_detection_d_b/dataloader_comparison
Clustering Experiments:src/components/distraction_detection_d_b/clustering_experiments
KL Results : src/components/distraction_detection_d_b/KL_results 
Variance Analysis: /notebooks/varience_analysis

Grid Search: src/components/distraction_detection_d_a/grid_search_ddp

Relevant Folder For Results:
Experiment 1 & 2: src/components/distraction_detection_d_a
Experminet 3 & 4: src/components/dinov2_linear
Plots Visualisation: notebooks/visualization_all

########## Datset Link: Drive and Act is publically Available at ####
Drive and Act is publically Available at: https://driveandact.com/

After downloading the video dataset, it can be converted into image dataset 
using the frame extraction scripts provided for respectively splits and videos.
 
############# Experimental Setup ###########

Conda environments required:
Name 1: vi_trans
name 2: dinov2

Packages installed in vi_trans: Follow the file: installed_packages_vi_trans_conda_environment.txt
Packages installed in dinov2: Follow the file: installed_packages_dinov2

# Dataloader based experiments are conducted using vi_trans environment
# Experiment 1 and 2 are conducted using vi_trans conda environment:
Activate it using:
conda activate vi_trans

# Used dinov2 environment for self-supervised learning based enccoder : DINOv2 vit_b_14 (Experiment 3 & Experiment 4)

Alternate for creating the dinov2 conda environment:
Follow the instructions for creating a conda environment for DINOv2 based experiments using following link:
(Link to DINOv2 github repository: https://github.com/facebookresearch/dinov2)
The training and evaluation code requires PyTorch 2.0 and xFormers 0.0.18 as well as a number of other 3rd party packages. 

# Create a conda environment using:
conda env create -f conda.yaml

# Activate the conda environment using:
conda activate dinov2

# Run the experiment using following command:
python path/to/the/module --config path/to/the/configurations

# An example to run the code:
python /home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/dinov2_linear/scripts_linear/dinov2_custom_linear.py --config /home/sur06423/hiwi/vit_exp/vision_tranformer_baseline/src/components/dinov2_linear/configs/dinov2_vitb_a_rgb_no_aug/DD_dinov2_VIT_01_SGD_0_0004_to_0_0002_split_0.txt

Similarly follow the same procedure for vit_b_16 based experiments.
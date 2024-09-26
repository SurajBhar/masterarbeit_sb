import os
import sys
import torch
# import matplotlib.pyplot as plt
import numpy as np
import pickle
# from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

# Local Imports
sys.path.append('/home/sur06423/hiwi/vit_exp/vision_tranformer_baseline')
# from src.components.Trainer_D_2.Dataloader_2.weighted_dataset import WeightedImageDataset

class DataLoaderB:
    def __init__(self, feature_file_path, label_file_path, img_path_file_path, num_categories):
        # self.batch_size = batch_size
        self.num_categories = num_categories
        self.features, self.labels, self.img_paths_list = self.load_data(feature_file_path, label_file_path, img_path_file_path)
        self.image_paths_all = [path for sublist in self.img_paths_list for path in sublist]
        self.dataloader_b, self.weights_list, self.all_labels, self.all_cluster_counts = self.initialize_dataloaders()

    def load_data(self, feature_file_path, label_file_path, img_path_file_path):
        with open(feature_file_path, 'rb') as file:
            features = pickle.load(file)
        with open(label_file_path, 'rb') as file:
            labels = pickle.load(file)
        with open(img_path_file_path, 'rb') as file:
            img_paths = pickle.load(file)
        return features, labels, img_paths

    def compute_weights_cosine_dist(self, features):
        cosine_dist_matrix = 1 - cosine_similarity(features).astype(np.float64)
        # Using Updated HDBSCAN for clustering with tuned Hyperparameters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=25, 
                                    min_samples=1, 
                                    cluster_selection_epsilon=0.0, 
                                    metric='precomputed', 
                                    cluster_selection_method='eom', 
                                    allow_single_cluster=False)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', cluster_selection_method='eom')
        labels = clusterer.fit_predict(cosine_dist_matrix)

        weights = np.zeros_like(labels, dtype=float)
        unique_labels = np.unique(labels)
        noise_label = -1
        # Initialize variables for managing the new outlier clusters
        max_label = labels.max()
        current_outlier_cluster_label = max_label + 1
        outlier_cluster_count = 0

        for label in unique_labels:
            if label == noise_label:
                # Process each noise point
                for noise_index in np.where(labels == noise_label)[0]:
                    # Assign it to the current outlier cluster
                    labels[noise_index] = current_outlier_cluster_label
                    outlier_cluster_count += 1
                    weights[noise_index] = 0.01  # Assign weight as 0.01

                    # If the outlier cluster reaches its max size, move to a new one
                    if outlier_cluster_count >= 50:
                        current_outlier_cluster_label += 1
                        outlier_cluster_count = 0
            else:
                # For non-noise points, distribute weights evenly within clusters
                indices = np.where(labels == label)[0]
                weights[indices] = 1.0 / len(indices)

        total_clusters = len(np.unique(labels)) - 1  # Exclude the original noise label

        return weights, labels, total_clusters

    def process_batches(self, dataloader):
        all_weights = []
        all_labels = []
        all_cluster_counts = []
        for batch_features in dataloader:
            weights, labels, total_clusters = self.compute_weights_cosine_dist(batch_features)
            all_weights.append(weights)
            all_labels.append(labels)
            all_cluster_counts.append(total_clusters)

        return all_weights, all_labels, all_cluster_counts
    
    def initialize_dataloaders(self):
        # Batch conversion of precomputed features, Batches = 254, Batch Size 1024, Feature size: [1,1280]
        # features_loader = [self.features[i:i+batch_size] for i in range(0, len(self.features), batch_size)]
        features_loader = [self.features[i:i+1024] for i in range(0, len(self.features), 1024)]
        gt_labels_loader = [self.labels[i:i+1024] for i in range(0, len(self.labels), 1024)]

        # Get the weights
        weights_list, all_labels, all_cluster_counts = self.process_batches(features_loader)
        # dataset_b = WeightedImageDataset(self.img_paths_list, weights_list, gt_labels_loader)
        # sampler_b = WeightedRandomSampler(dataset_b.weights, num_samples=len(dataset_b.weights), replacement=True)
        # dataloader_b = DataLoader(dataset_b, batch_size=self.batch_size, sampler=sampler_b, num_workers=10)

        return gt_labels_loader, weights_list, all_labels, all_cluster_counts 
    
# Create an instance of DataLoaderB class
# data_loader_instance = DataLoaderB(feature_file_path, label_file_path, img_path_file_path, num_categories, batch_size)

# Accessing the attributes
# dataloader_b = data_loader_instance.dataloader_b

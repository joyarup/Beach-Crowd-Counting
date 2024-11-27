# src/config/default_params.py
import os
import yaml

class Config:  # Changed from Config to config
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.dataset_folder = self.config['paths']['dataset_folder']
        self.output_folder = self.config['paths']['output_folder']
        self.ground_truth_file = self.config['paths']['ground_truth_file']
        
        # Detection parameters
        self.detection = self.config['detection_params']
        
        # Background parameters
        self.background = self.config['background_params']
        
        # Evaluation parameters
        self.evaluation = self.config['evaluation_params']
        
        # Visualization parameters
        self.visualization = self.config['visualization_params']
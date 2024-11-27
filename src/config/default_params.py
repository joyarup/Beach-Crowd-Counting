# src/config/default_params.py
import os
import yaml
from pathlib import Path

class Config:
    def __init__(self):
        # Get the directory containing this script
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.yaml"
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths relative to project root
        project_root = config_dir.parent.parent
        self.dataset_folder = project_root / self.config['paths']['dataset_folder']
        self.output_folder = project_root / self.config['paths']['output_folder']
        self.ground_truth_file = project_root / self.config['paths']['ground_truth_file']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Other parameters
        self.detection = self.config['detection_params']
        self.background = self.config['background_params']
        self.evaluation = self.config['evaluation_params']
        self.visualization = self.config['visualization_params']
    
    def get_detection_param(self, *keys):
        """Get nested detection parameters"""
        return self._get_nested_param(self.detection, keys)
    
    def get_background_param(self, *keys):
        """Get nested background parameters"""
        return self._get_nested_param(self.background, keys)
    
    def get_evaluation_param(self, *keys):
        """Get nested evaluation parameters"""
        return self._get_nested_param(self.evaluation, keys)
    
    def get_visualization_param(self, *keys):
        """Get nested visualization parameters"""
        return self._get_nested_param(self.visualization, keys)
    
    def _get_nested_param(self, params, keys):
        """Helper method to get nested parameters"""
        for key in keys:
            params = params.get(key)
            if params is None:
                return None
        return params
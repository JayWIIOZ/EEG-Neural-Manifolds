import numpy as np

class Config:
    
    def __init__(self, ao: bool):
        self.ao = ao
        
        """Configuration parameters for the analysis"""
        self.ROIS = [1, 2, 19, 20, 59, 60, 61, 62]
        self.ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
        self.PARADIGMS = ['AO', 'rest']
        self.FREQUENCY_BANDS = ['alpha', 'beta', 'theta', 'delta']
        # REST_TRIALS = 13
        self.RANDOM_SEED = 24
        
        # Paths
        self.ROOT_PATH = '/dataset/acute_storke/RESULTS/'
        if self.ao:
            self.PATH = '/dataset/acute_storke/RESULTS/voxel_npy/'
        else:
            self.PATH = '/dataset/acute_storke/RESULTS/rest_npy/'


class DataLoader:
    """Handles data loading and preprocessing"""
    def __init__(self, config: Config):
        self.config = config
        
    def load_subject_data(self, subj: str, roi: int, freq: str, ao: bool) -> np.ndarray:
        """Load data for a single subject and ROI"""
        data_path = self.config.ROOT_PATH / subj / f'roi_{roi}'
        if ao:
            file_name = f"{subj}_{self.config.PARADIGM[0]}_pca_trial_{freq}.npy"
        else:
            file_name = f"{subj}_{self.config.PARADIGM[1]}_pca_trial_{freq}.npy"
        try:
            return np.load(data_path / file_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_name}")
        
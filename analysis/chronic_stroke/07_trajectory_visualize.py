import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from utils import canoncorr, divide_pair, get_colors
import os
import logging
from tqdm import tqdm

class Config:
    """Configuration parameters"""
    # ROIS = [1, 2, 19, 20, 59, 60, 61, 62]
    ROIS = [1]
    ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
    TRAIN_STAGES = ['pre', 'post']
    PARADIGM = 'AO1'
    FREQ_BAND = 'alpha'
    SUBJECTS = ['kmt', 'ock']
    DATA_PATH = 'chronic_stroke/pca_data/'
    SAVE_PATH = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/trajectory/'

class DynamicsAligner:
    def __init__(self, config):
        self.cfg = config
        os.makedirs(self.cfg.SAVE_PATH, exist_ok=True)
        
    def load_data(self, subject, roi, stage):
        """Load PCA data for a subject"""
        try:
            data_path = os.path.join(
                self.cfg.DATA_PATH, stage, self.cfg.PARADIGM,
                subject, str(roi),
                f"{subject}_pca_trial_{self.cfg.FREQ_BAND}.npy"
            )
            return np.load(data_path)
        except Exception as e:
            logging.error(f"Error loading data for {subject}: {str(e)}")
            return None

    def preprocess_data(self, data_list):
        """Preprocess data to same dimensions"""
        processed_data = []
        for data in data_list:
            if data is None:
                continue
            rank = min(np.linalg.matrix_rank(data))
            processed_data.append(data[:, :, :rank].reshape(-1, rank))
            
        if not processed_data:
            return None
            
        # Standardize dimensions
        time_min = min(d.shape[0] for d in processed_data)
        rank_min = min(d.shape[1] for d in processed_data)
        return [d[:time_min, :rank_min] for d in processed_data]

    def align_dynamics(self, data_list, time_len):
        """Align dynamics across subjects"""
        # Get subject pairs
        subject_pairs = divide_pair(range(len(data_list)))
        aligned_data = []
        cca_scores = []
        
        for pair in subject_pairs:
            # Compute CCA
            A, B, r, *_ = canoncorr(data_list[pair[0]], data_list[pair[1]], fullReturn=True)
            
            # SVD decomposition
            U1, _, Vh1 = svd(A, full_matrices=False)
            U2, _, Vh2 = svd(B, full_matrices=False)
            
            # Transform data
            data1 = data_list[pair[0]].reshape(-1, time_len, r.shape[-1]) @ U1 @ Vh1
            data2 = data_list[pair[1]].reshape(-1, time_len, r.shape[-1]) @ U2 @ Vh2
            
            aligned_data.append([data1, data2])
            cca_scores.append(r)
            
        return aligned_data, cca_scores

    def visualize_alignment(self, aligned_data, subject_pairs, roi, stage):
        """Visualize aligned dynamics"""
        colors = get_colors(8, colormap='Paired')
        
        for pair_idx, pair_data in enumerate(aligned_data):
            # Calculate mean trajectories
            mean_trajectories = [np.mean(data, axis=0) for data in pair_data]
            
            # Create plot
            fig = plt.figure()
            ax = plt.axes(projection='3d', fc='None')
            
            for i, traj in enumerate(mean_trajectories):
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       color=colors[-2*(i+1)])
                       
            # Customize plot
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('CC1', fontsize=15)
            ax.set_ylabel('CC2', fontsize=15)
            ax.set_zlabel('CC3', fontsize=15)
            
            # Set title and save
            subj1, subj2 = subject_pairs[pair_idx]
            ax.set_title(f"{self.cfg.SUBJECTS[subj1]} x {self.cfg.SUBJECTS[subj2]}", 
                        fontsize=15)
            
            save_name = (f"{self.cfg.SUBJECTS[subj1]}-{self.cfg.SUBJECTS[subj2]}"
                        f"_Region_{roi}_{self.cfg.FREQ_BAND}_{stage}_aligned.png")
            fig.savefig(os.path.join(self.cfg.SAVE_PATH, save_name), 
                       format='png', dpi=1000)
            plt.close()

def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
                       
    config = Config()
    aligner = DynamicsAligner(config)
    
    for stage in tqdm(config.TRAIN_STAGES, desc="Processing stages"):
        for roi in tqdm(config.ROIS, desc=f"Processing ROIs for {stage}"):
            # Load data
            data_list = []
            for subject in config.SUBJECTS:
                data = aligner.load_data(subject, roi, stage)
                if data is not None:
                    data_list.append(data)
                    
            if not data_list:
                continue
                
            # Process data
            processed_data = aligner.preprocess_data(data_list)
            if processed_data is None:
                continue
                
            # Align dynamics
            time_len = data_list[0].shape[1]
            aligned_data, _ = aligner.align_dynamics(processed_data, time_len)
            
            # Visualize results
            subject_pairs = divide_pair(range(len(processed_data)))
            aligner.visualize_alignment(aligned_data, subject_pairs, roi, stage)

if __name__ == "__main__":
    main()
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from jay_code.utils import canoncorr, divide_pair, get_colors

class Config:
    """Configuration parameters for low dimensional alignment analysis"""
    def __init__(self):
        # Analysis parameters
        self.ROIS: List[int] = [1, 2, 19, 20, 59, 60, 61, 62]
        self.ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
        self.PARADIGM: str = 'AO'
        self.FREQ_BAND: str = ['alpha', 'beta', 'theta', 'delta']
        self.TRAIN_STAGES: List[str] = ['high', 'low']
        self.RANDOM_SEED: int = 24
        
        # File paths
        self.CSV_PATH: Path = Path('/code/dataset/acute_stroke_dataset/participants_copy.csv')
        self.DATA_PATH: Path = Path('/code/dataset/acute_stroke_dataset/RESULTS/voxel_npy/')
        self.SAVE_PATH: Path = Path('/code/analysis/results/manifolds/')


class DataProcessor:
    """Handles data loading and preprocessing"""
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.RANDOM_SEED)

    def get_subject_groups(self) -> Tuple[List[str], List[str]]:
        """Extract high and low NIHSS score groups"""
        df = pd.read_csv(self.config.CSV_PATH)
        high = df[df['NIHSS'] >= 6]['Participant_ID'].tolist()
        low = df[df['NIHSS'] <= 2]['Participant_ID'].tolist()
        return high, low

    def load_subject_data(self, subject: str, roi: int, stage: str) -> np.ndarray:
        """Load and preprocess subject data"""
        data_path = (self.config.DATA_PATH / stage / subject / 
                    f'roi_{roi}' / f'{subject}_{self.config.PARADIGM}_pca_trial_{self.config.FREQ_BAND}.npy')
        
        try:
            data = np.load(data_path)
            rank = min(np.linalg.matrix_rank(data))
            return np.reshape(data[:, :, :rank], (-1, rank))
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")


class CCAAnalyzer:
    """Performs CCA analysis between subject pairs"""
    @staticmethod
    def align_subjects(data1: np.ndarray, data2: np.ndarray, time_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Align data between two subjects using CCA"""
        A1, B1, r1, *_ = canoncorr(data1, data2, fullReturn=True)
        U1, _, Vh1 = svd(A1, full_matrices=False)
        U2, _, Vh2 = svd(B1, full_matrices=False)
        
        aligned1 = np.reshape(data1, (-1, time_len, r1.shape[-1])) @ U1 @ Vh1
        aligned2 = np.reshape(data2, (-1, time_len, r1.shape[-1])) @ U2 @ Vh2
        return aligned1, aligned2
    
    
class Visualizer:
    """Handles 3D visualization of aligned data"""
    def __init__(self, config: Config):
        self.config = config
        self.colors = get_colors(8, colormap='Set3')

    def plot_alignment(self, data1: np.ndarray, data2: np.ndarray, 
                      subj1: str, subj2: str, roi: int, stage: str) -> None:
        """Plot aligned data in 3D space"""
        fig = plt.figure()
        ax = plt.axes(projection='3d', fc='None')
        
        # Plot averaged trajectories
        ax.plot(data1[:, 0], data1[:, 1], data1[:, 2], color=self.colors[0])
        ax.plot(data2[:, 0], data2[:, 1], data2[:, 2], color=self.colors[2])
        
        # Configure plot
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('CC1', fontsize=15)
        ax.set_ylabel('CC2', fontsize=15)
        ax.set_zlabel('CC3', fontsize=15)
        ax.set_title(f'{subj1} x {subj2}', fontsize=15)
        
        # Save figure
        fig.savefig(self.config.SAVE_PATH / 
                   f'{subj1}-{subj2}_Region_{roi}_{self.config.FREQ_BAND}_{stage}_aligned.eps',
                   format='eps', dpi=1000)
        plt.close(fig)

def main():
    """Main execution function"""
    config = Config()
    processor = DataProcessor(config)
    analyzer = CCAAnalyzer()
    visualizer = Visualizer(config)

    # Get subject groups
    high_group, low_group = processor.get_subject_groups()
    subject_groups = [high_group, low_group]

    # Process each ROI
    for roi in config.ROIS:
        for stage_idx, stage in enumerate(config.TRAIN_STAGES):
            # Load subject data
            subject_data = []
            for subject in subject_groups[stage_idx]:
                try:
                    data = processor.load_subject_data(subject, roi, stage)
                    subject_data.append(data)
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    continue

            # Process subject pairs
            time_len = subject_data[0].shape[0]
            subject_pairs = divide_pair(subject_data)

            # Analyze and visualize specific pairs
            for pair_idx in range(14, 26):
                pair = subject_pairs[pair_idx]
                aligned1, aligned2 = analyzer.align_subjects(
                    subject_data[pair[0]], 
                    subject_data[pair[1]], 
                    time_len
                )

                # Average and visualize
                avg1 = np.mean(aligned1, axis=0)
                avg2 = np.mean(aligned2, axis=0)
                
                visualizer.plot_alignment(
                    avg1, avg2,
                    subject_groups[stage_idx][pair[0]],
                    subject_groups[stage_idx][pair[1]],
                    roi, stage
                )

if __name__ == "__main__":
    main()
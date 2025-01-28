import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import pickle
from utils import *
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
ROIS = [1, 2, 19, 20, 59, 60, 61, 62]
TRAIN_STAGES = ['pre', 'post']
PARADIGM = 'AO1'
FREQ_BANDS = ['theta', 'alpha', 'beta']
FS = 100
NPERSEG_GROUP = np.arange(25, 101, 5)
FREQ_RANGES = {
    'delta': [1, 4],
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta': [12, 30],
    'gamma': [30, 40]
}
SUBJECTS = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wsc', 'wwf']

class PSDAnalyzer:
    def __init__(self, load_path):
        self.load_path = load_path
        
    def load_voxel_data(self, subj, roi, train_stage, trial_num):
        """Load voxel data for a subject"""
        hemisphere = 'l' if roi % 2 == 0 else 'r'
        file_path = f"{self.load_path}/{subj}/trial/{roi}/{subj}_{PARADIGM}_{train_stage}_voxel_{trial_num}_{hemisphere}.mat"
        return loadmat(file_path)['momint_1'][:, :200]
    
    def compute_psd(self, data, freq_band, nperseg, noverlap):
        """Compute PSD for given data"""
        f, _, Zxx = signal.stft(data, FS, nperseg=nperseg, noverlap=noverlap, scaling='psd')
        mask = np.logical_and(f >= FREQ_RANGES[freq_band][0], f <= FREQ_RANGES[freq_band][1])
        psd = np.mean(Zxx[:, mask, :].real, axis=1)
        return smooth_average(psd, 3, 3)[:, 10:40]

def main():
    logging.info("Starting PSD analysis...")
    analyzer = PSDAnalyzer('F:/CUHK_intern/RESULTS/Multimodality/')
    
    cca_scores = []
    for freq_band in tqdm(FREQ_BANDS, desc="Processing frequency bands"):
        stage_scores = []
        for stage in tqdm(TRAIN_STAGES, desc=f"Processing stages for {freq_band}"):
            nperseg_scores = []
            for nperseg in tqdm(NPERSEG_GROUP, desc=f"Processing window lengths for {stage}"):
                noverlap = nperseg - 1
                roi_scores = []
                
                for roi in tqdm(ROIS, desc="Processing ROIs"):
                    subj_scores = []
                    trial_min = float('inf')
                    
                    # Get minimum trial count
                    for subj in SUBJECTS:
                        trial_count = len([f for f in os.listdir(f"{analyzer.load_path}/{subj}/trial") 
                                         if f.endswith('.mat')])
                        trial_min = min(trial_min, trial_count)
                    
                    for subj in SUBJECTS:
                        try:
                            # Load and process data
                            data_pca = np.load(f"{analyzer.load_path}/{stage}/{PARADIGM}/{subj}/trial/{roi}/"
                                             f"{subj}_{PARADIGM}_{stage}_pca_trial_{freq_band}.npy")
                            
                            voxel_data = [analyzer.load_voxel_data(subj, roi, stage, n) 
                                         for n in range(1, trial_min + 1)]
                            
                            psd_list = [analyzer.compute_psd(data, freq_band, nperseg, noverlap) 
                                       for data in voxel_data]
                            
                            # Compute CCA scores
                            psd_pca, _ = get_data_mat(psd_list, 20)
                            rank_min = min(np.linalg.matrix_rank(psd_pca), 
                                         np.linalg.matrix_rank(data_pca))
                            
                            psd_reshaped = np.reshape(psd_pca[:, :, :rank_min], (-1, rank_min))
                            data_reshaped = np.reshape(data_pca[:trial_min, :, :rank_min], (-1, rank_min))
                            
                            r = canoncorr(data_reshaped, psd_reshaped, fullReturn=False)
                            subj_scores.append(np.mean(r[:4]))
                            
                        except Exception as e:
                            logging.error(f"Error processing subject {subj}: {str(e)}")
                            continue
                            
                    roi_scores.append(subj_scores)
                nperseg_scores.append(roi_scores)
            stage_scores.append(nperseg_scores)
        cca_scores.append(stage_scores)
    
    # Save results
    logging.info("Saving results...")
    with open('F:/CUHK_Intern/RESULTS/Multimodality/ManiPSDNperseg.pkl', 'wb') as file:
        pickle.dump(cca_scores, file)
    
    # Visualization
    logging.info("Creating visualizations...")
    plot_results(cca_scores, TRAIN_STAGES, FREQ_BANDS, PARADIGM)
    logging.info("Analysis completed!")

def plot_results(data, stages, freq_bands, paradigm):
    """Plot analysis results"""
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))
    
    for stage_idx, stage in enumerate(stages):
        fig, ax = plt.subplots()
        stage_data = data[:, stage_idx, :, :]
        
        for band_idx, band in enumerate(freq_bands):
            shaded_errorbar(ax, NPERSEG_GROUP, stage_data[band_idx, :, :], label=band)
            
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Length of Sliding Window', fontsize=15)
        ax.set_ylabel('Canonical Correlation', fontsize=15)
        ax.set_xticks(NPERSEG_GROUP)
        ax.set_title(f"{paradigm}-{stage}", fontsize=15)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        fig.savefig(f'F:/CUHK_Intern/RESULTS/figure/Multimodality/ManiPSDWindowLength_{paradigm}_{stage}.eps',
                    format='eps', dpi=1000)
        plt.close()

if __name__ == "__main__":
    main()
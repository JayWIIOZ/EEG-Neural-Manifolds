"""
Integrated and optimized code for PSD-CCA analysis with enhanced readability and modularity.
Combines functionalities from ManifoldPsdAlign.py and ManiPSDwithNperseg.py.
Supports variable window lengths, multiple frequency bands, and cross-subject analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import pickle
from utils import smooth_average, get_data_mat, canoncorr, shaded_errorbar
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global constants
ROIS = [1, 2, 19, 20, 59, 60, 61, 62]
ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
TRAIN_STAGES = ['pre', 'post']
PARADIGM = 'AO1'
FREQ_BANDS = ['theta', 'alpha', 'beta']
FS = 100
NPERSEG_GROUP = np.arange(25, 101, 5)  # Window lengths to test
FREQ_RANGES = {
    'delta': [1, 4],
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta': [12, 30],
    'gamma': [30, 40]
}
SUBJECTS = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wsc', 'wwf']
BASE_PATH = 'F:/CUHK_intern/RESULTS/Multimodality/'

class PSDAnalyzer:
    """Core class for PSD and CCA analysis with modular components"""
    
    def __init__(self, mode='fixed_window'):
        """
        Initialize analyzer with operation mode
        :param mode: 'fixed_window' (original) or 'variable_window' (nperseg testing)
        """
        self.mode = mode
        self._validate_mode()

    def _validate_mode(self):
        """Ensure valid operation mode"""
        if self.mode not in ['fixed_window', 'variable_window']:
            raise ValueError("Invalid mode. Choose 'fixed_window' or 'variable_window'")

    def load_voxel_data(self, subj, roi, stage, trial_num):
        """
        Load voxel data from MAT files
        :return: Numpy array of shape (timepoints, features)
        """
        hemisphere = 'l' if roi % 2 == 0 else 'r'
        path = f"{BASE_PATH}/{stage}/{PARADIGM}/{subj}/trial/{roi}/{subj}_{PARADIGM}_{stage}_voxel_{trial_num}_{hemisphere}.mat"
        return loadmat(path)['momint_1'][:, :200]

    def compute_psd(self, data, freq_band, nperseg=25, noverlap=24):
        """
        Compute power spectral density using STFT
        :return: Smoothed PSD array of shape (features, timepoints)
        """
        f, _, Zxx = signal.stft(data, FS, nperseg=nperseg, noverlap=noverlap, scaling='psd')
        mask = np.logical_and(f >= FREQ_RANGES[freq_band][0], f <= FREQ_RANGES[freq_band][1])
        psd = np.mean(Zxx[:, mask, :].real, axis=1)
        return smooth_average(psd, 3, 3)[:, 10:40]

    def run_analysis(self):
        """Main analysis pipeline controller"""
        logging.info("Starting analysis in %s mode", self.mode)
        
        if self.mode == 'fixed_window':
            self._fixed_window_analysis()
        else:
            self._variable_window_analysis()

    def _fixed_window_analysis(self):
        """Original analysis with fixed window size"""
        cca_results = []
        for stage in tqdm(TRAIN_STAGES, desc="Stages"):
            stage_results = []
            for roi in tqdm(ROIS, desc=f"ROIs ({stage})"):
                roi_results = []
                trial_min = self._get_min_trials(stage)
                
                for subj in tqdm(SUBJECTS, desc="Subjects"):
                    try:
                        data_pca, psd_pca = self._process_subject(subj, roi, stage, trial_min)
                        scores = self._compute_cca(data_pca, psd_pca, trial_min)
                        roi_results.append(scores)
                    except Exception as e:
                        logging.warning("Subject %s failed: %s", subj, str(e))
                stage_results.append(roi_results)
            self._plot_stage_results(stage_results, stage)
            cca_results.append(stage_results)
        return cca_results

    def _variable_window_analysis(self):
        """Window length sensitivity analysis"""
        results = []
        for freq_band in tqdm(FREQ_BANDS, desc="Frequency Bands"):
            band_results = []
            for stage in tqdm(TRAIN_STAGES, desc="Stages"):
                stage_results = []
                for nperseg in tqdm(NPERSEG_GROUP, desc="Window Lengths"):
                    noverlap = nperseg - 1
                    roi_scores = []
                    
                    for roi in ROIS:
                        trial_min = self._get_min_trials(stage)
                        subj_scores = []
                        
                        for subj in SUBJECTS:
                            try:
                                data_pca, psd_pca = self._process_subject(
                                    subj, roi, stage, trial_min, 
                                    freq_band, nperseg, noverlap
                                )
                                scores = self._compute_cca(data_pca, psd_pca, trial_min)
                                subj_scores.append(np.mean(scores[:4]))
                            except Exception as e:
                                logging.warning("Subject %s failed: %s", subj, str(e))
                        roi_scores.append(subj_scores)
                    stage_results.append(roi_scores)
                band_results.append(stage_results)
            results.append(band_results)
        
        self._save_results(results)
        self._plot_window_results(results)
        return results

    def _process_subject(self, subj, roi, stage, trial_min, 
                        freq_band='theta', nperseg=25, noverlap=24):
        """Process data for a single subject"""
        # Load PCA data
        data_pca = np.load(
            f"{BASE_PATH}/{stage}/{PARADIGM}/{subj}/trial/{roi}/"
            f"{subj}_{PARADIGM}_{stage}_pca_trial_{freq_band}.npy"
        )
        
        # Process voxel data
        voxel_data = [self.load_voxel_data(subj, roi, stage, n) 
                     for n in range(1, trial_min + 1)]
        psd_list = [self.compute_psd(data, freq_band, nperseg, noverlap) 
                   for data in voxel_data]
        psd_pca, _ = get_data_mat(psd_list, 20)
        
        return data_pca, psd_pca

    def _compute_cca(self, data_pca, psd_pca, trial_min):
        """Perform Canonical Correlation Analysis"""
        rank_min = min(np.linalg.matrix_rank(psd_pca), 
                      np.linalg.matrix_rank(data_pca))
        psd_flat = np.reshape(psd_pca[:, :, :rank_min], (-1, rank_min))
        data_flat = np.reshape(data_pca[:trial_min, :, :rank_min], (-1, rank_min))
        return canoncorr(data_flat, psd_flat, fullReturn=False)

    def _get_min_trials(self, stage):
        """Get minimum number of trials across subjects"""
        return min(len(os.listdir(f"{BASE_PATH}/{stage}/{PARADIGM}/{subj}/trial")) 
                 for subj in SUBJECTS)

    def _plot_stage_results(self, data, stage):
        """Plot results for fixed window analysis"""
        fig, ax = plt.subplots()
        for roi_idx, roi_data in enumerate(data):
            Y = np.mean(roi_data, axis=0)
            ax.plot(np.arange(1, len(Y)+1), Y, label=ROI_LABELS[roi_idx])
        
        ax.legend(fontsize=10)
        ax.set(xlabel='Dimensions', ylabel='Canonical Correlation',
              title=f'{PARADIGM}-{stage}', ylim=[0, 1.01])
        fig.savefig(f'{BASE_PATH}/figure/ManiPsd_{PARADIGM}_{stage}.eps', 
                   format='eps', dpi=1000)
        plt.close()

    def _plot_window_results(self, data):
        """Plot window length sensitivity results"""
        data = np.array(data)
        for stage_idx, stage in enumerate(TRAIN_STAGES):
            fig, ax = plt.subplots()
            for band_idx, band in enumerate(FREQ_BANDS):
                stage_data = data[band_idx, stage_idx]
                shaded_errorbar(ax, NPERSEG_GROUP, stage_data, label=band)
            
            ax.legend(fontsize=12)
            ax.set(xlabel='Window Length', ylabel='Canonical Correlation',
                  title=f'{PARADIGM}-{stage}', ylim=[0, 1])
            fig.savefig(f'{BASE_PATH}/figure/ManiPSDWindow_{PARADIGM}_{stage}.eps',
                       format='eps', dpi=1000)
            plt.close()

    def _save_results(self, data):
        """Save analysis results to pickle file"""
        with open(f'{BASE_PATH}/ManiPSDResults.pkl', 'wb') as f:
            pickle.dump(data, f)
            logging.info("Results saved to pickle file")

if __name__ == "__main__":
    analyzer = PSDAnalyzer(mode='variable_window')  # Switch modes here
    analyzer.run_analysis()
    logging.info("Analysis completed successfully")
import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from utils import smooth_average, get_data_mat, canoncorr
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
ROIS = [1, 2, 19, 20, 59, 60, 61, 62]
ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
TRAIN_STAGES = ['pre', 'post']
PARADIGM = 'AO1'
FREQ_BAND = 'theta'
SUBJ_LIST = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wsc', 'wwf']

# Signal processing parameters
FS = 100
NPERSEG = 25
NOVERLAP = 24
FREQ_BANDS = {
    'delta': [1, 4],
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta': [12, 30],
    'gamma': [30, 40]
}

def load_voxel_data(load_path: str, subj: str, roi: int, trial_num: int, stage) -> np.ndarray:
    """Load voxel data for given subject and ROI"""
    side = 'l' if roi % 2 == 0 else 'r'
    file_path = f"{load_path}/{subj}/trial/{roi}/{subj}_{PARADIGM}_{stage}_voxel_{trial_num}_{side}.mat"
    return loadmat(file_path)['momint_1'][:, :200]

def compute_psd(data: np.ndarray, freq_band: list) -> np.ndarray:
    """Compute PSD for given data and frequency band"""
    f, _, Zxx = signal.stft(data, FS, nperseg=NPERSEG, noverlap=NOVERLAP, scaling='psd')
    mask = np.logical_and(f >= freq_band[0], f <= freq_band[1])
    psd = np.mean(Zxx[:, mask, :].real, axis=1)
    return smooth_average(psd, 3, 3)[:, 10:40]

def compute_cca_scores(psd_pca: np.ndarray, data_pca: np.ndarray, trial_min: int) -> np.ndarray:
    """Compute CCA scores between PSD and PCA data"""
    rank_min = min(np.linalg.matrix_rank(psd_pca), np.linalg.matrix_rank(data_pca))
    psd_reshaped = np.reshape(psd_pca[:, :, :rank_min], (-1, rank_min))
    data_reshaped = np.reshape(data_pca[:trial_min, :, :rank_min], (-1, rank_min))
    return canoncorr(data_reshaped, psd_reshaped, fullReturn=False)

def plot_results(cca_scores: list, stage: str):
    """Plot CCA results for given stage"""
    fig, ax = plt.subplots()
    for i, scores in enumerate(cca_scores):
        Y = np.array(scores)
        ax.plot(np.arange(1, np.mean(Y, axis=0).shape[0]+1), 
                np.mean(Y, axis=0), 
                label=ROI_LABELS[i])
    
    ax.legend(fontsize=10)
    ax.set_xlabel('Dimensions', fontsize=15)
    ax.set_ylabel('Canonical Correlation', fontsize=15)
    ax.set_title(f'{PARADIGM}-{FREQ_BAND}-{stage.capitalize()}', fontsize=15)
    ax.set_ylim([0, 1.01])
    ax.set_xticks(np.arange(0, 19, 2))
    fig.tight_layout()
    
    save_path = f'F:/CUHK_Intern/RESULTS/figure/Multimodality/ManiPsd_{PARADIGM}_{FREQ_BAND}_{stage}.eps'
    fig.savefig(save_path, format='eps', dpi=1000)
    plt.show()

def main():
    logging.info("Starting manifold PSD alignment analysis...")
    cca_scores = []
    
    for stage in tqdm(TRAIN_STAGES, desc="Processing stages"):
        stage_scores = []
        load_path = f'F:/CUHK_intern/RESULTS/Multimodality/{stage}/{PARADIGM}/'
        logging.info(f"Processing stage: {stage}")
        
        for roi in tqdm(ROIS, desc=f"Processing ROIs for {stage}"):
            roi_scores = []
            try:
                trial_min = min(len([f for f in os.listdir(f"{load_path}/{subj}/trial") 
                                   if f.endswith('.mat')]) for subj in SUBJ_LIST)
                
                for subj in tqdm(SUBJ_LIST, desc=f"Processing subjects for ROI {roi}"):
                    try:
                        # Process voxel data
                        voxel_data = [load_voxel_data(load_path, subj, roi, n, stage) 
                                     for n in range(1, trial_min + 1)]
                        psd_data = [compute_psd(data, FREQ_BANDS[FREQ_BAND]) 
                                   for data in voxel_data]
                        
                        # Compute PCA and CCA
                        psd_pca, _ = get_data_mat(psd_data, 20)
                        data_pca = np.load(f"{load_path}/{subj}/trial/{roi}/{subj}_{PARADIGM}_{stage}_pca_trial_{FREQ_BAND}.npy")
                        
                        scores = compute_cca_scores(psd_pca, data_pca, trial_min)
                        roi_scores.append(scores)
                        logging.info(f"Completed processing subject {subj}")
                        
                    except Exception as e:
                        logging.error(f"Error processing subject {subj}: {str(e)}")
                        continue
                        
                stage_scores.append(roi_scores)
                logging.info(f"Completed processing ROI {roi}")
                
            except Exception as e:
                logging.error(f"Error processing ROI {roi}: {str(e)}")
                continue
                
        cca_scores.append(stage_scores)
        
        # Plot results for current stage
        logging.info(f"Creating visualization for stage {stage}")
        plot_results(stage_scores, stage)
        
    logging.info("Analysis completed successfully!")

if __name__ == "__main__":
    main()
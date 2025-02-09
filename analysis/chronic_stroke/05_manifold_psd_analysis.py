import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from utils import *
import logging

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class Config:
    """Configuration parameters for PSD analysis"""
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]  
    train_stage = ['pre', 'post']
    Paradigm = 'AO1'
    freqb = 'alpha'
    subj_list = ['kmt', 'ock']
    fs = 100  # Sampling frequency (Hz)
    nperseg = 25  
    noverlap = 24
    freqBand = {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 40]
    }
    mat_path = 'chronic_stroke/'
    pca_path = 'chronic_stroke/pca_data/'
    save_path = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/manifold_psd/'
    trial_min = 13
    ROIs_label = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 
                  'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']

def load_voxel_data(config, load_path, subj, roi, train_stage):
    """Load voxel data for analysis"""
    data_voxel_list = []
    suffix = '_r' if roi % 2 != 0 else '_l'
    
    logging.info(f"Loading data for subject {subj}, ROI {roi}")
    for num in range(1, config.trial_min + 1):
        mat_path = os.path.join(
            load_path, subj, str(roi),
            f'{subj}_{config.Paradigm}_{train_stage}_voxel_{num}{suffix}.mat'
        )
        try:
            data_voxel = loadmat(mat_path)['momint_1']
            data_voxel_list.append(data_voxel[:, :200])
        except Exception as e:
            logging.error(f"Failed to load {mat_path}: {e}")
            raise
            
    return data_voxel_list

def process_psd(data_voxel_list, config):
    """Calculate PSD for voxel data"""
    voxel_psd_list = []
    for trial_ii in range(config.trial_min):
        f, t, Zxx = signal.stft(
            data_voxel_list[trial_ii], 
            config.fs, 
            nperseg=config.nperseg, 
            noverlap=config.noverlap, 
            scaling='psd'
        )
        freq_mask = np.logical_and(
            f >= config.freqBand[config.freqb][0], 
            f <= config.freqBand[config.freqb][1]
        )
        voxel_psd = np.mean(Zxx[:, freq_mask, :].real, axis=1)
        voxel_psd_ = smooth_average(voxel_psd, 3, 3)[:, 10:40]
        voxel_psd_list.append(voxel_psd_)
    return voxel_psd_list

def plot_results(config, data, stage_idx, stage_name):
    """Plot analysis results for a given stage"""
    fig, ax = plt.subplots(ncols=1)
    for i in range(len(data[stage_idx])):
        Y = np.array(data[stage_idx][i])
        ax.plot(
            np.arange(1, np.mean(Y, axis=0).shape[0] + 1), 
            np.mean(Y, axis=0), 
            label=str(config.ROIs_label[i])
        )
        
    ax.legend(fontsize=10)
    ax.set_xlabel('Dimensions', fontdict={'size': 15})
    ax.set_ylabel('Canonical Correlation', fontdict={'size': 15})
    ax.set_title(f"{config.Paradigm}-{config.freqb}-{stage_name}", 
                fontdict={'size': 15})
    ax.set_ylim([0, 1.01])
    ax.set_xticks(np.arange(0, 19, 2))
    fig.tight_layout()
    
    save_path = os.path.join(
        config.save_path,
        f'ManiPsd_{config.Paradigm}_{config.freqb}_{stage_name}.png'
    )
    fig.savefig(save_path, format='png', dpi=1000)
    logging.info(f"Saved plot to {save_path}")
    plt.show()

def analyze_stage_data(config, train_stage):
    """Analyze data for a specific training stage"""
    logging.info(f"Processing {train_stage} stage")
    CCA_score_stage = []
    load_path = os.path.join(config.mat_path, train_stage, config.Paradigm)

    for roi in config.ROIs:
        logging.info(f"Processing ROI {roi}")
        CCA_score_roi = []

        for subj in config.subj_list:
            # Load and process data
            data_voxel_list = load_voxel_data(
                config, load_path, subj, roi, train_stage
            )
            pca_path = os.path.join(
                config.pca_path, train_stage, config.Paradigm,
                subj, str(roi), f'{subj}_pca_trial_{config.freqb}.npy'
            )
            data_pca = np.load(pca_path)

            # Calculate PSD
            voxel_psd_list = process_psd(data_voxel_list, config)

            # Perform PCA and CCA
            psd_pca, _, _ = get_data_mat(voxel_psd_list, 20)
            rank_min = min(
                min(np.linalg.matrix_rank(psd_pca)), 
                min(np.linalg.matrix_rank(data_pca))
            )

            psd_pca_reshape = np.reshape(
                np.array(psd_pca)[:, :, :rank_min], (-1, rank_min)
            )
            data_pca_reshape = np.reshape(
                data_pca[:config.trial_min, :, :rank_min], (-1, rank_min)
            )

            r = canoncorr(data_pca_reshape, psd_pca_reshape, fullReturn=False)
            CCA_score_roi.append(r)

        dim_min = min([score.shape[0] for score in CCA_score_roi])
        CCA_score_stage.append([temp[:dim_min] for temp in CCA_score_roi])

    return CCA_score_stage

def main():
    """Main execution flow"""
    setup_logging()
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    
    logging.info("Starting PSD manifold analysis")
    CCA_score = []

    # Process each training stage
    for stage in config.train_stage:
        CCA_score_stage = analyze_stage_data(config, stage)
        CCA_score.append(CCA_score_stage)

    # Plot results
    plot_results(config, CCA_score, 0, 'Pre')
    plot_results(config, CCA_score, 1, 'Post')
    
    logging.info("Analysis completed")

if __name__ == '__main__':
    main()
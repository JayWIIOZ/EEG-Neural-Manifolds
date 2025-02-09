import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import pickle
from utils import *
import logging

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class Config:
    """Configuration parameters for the analysis"""
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]  
    train_stage = ['pre', 'post']  
    Paradigm = 'AO1'  
    freqb_list = ['alpha']  
    subj_list = ['kmt', 'ock']  
    fs = 100  # Sampling frequency (Hz)
    nperseg_group = np.arange(25,101,5)  
    freqBand = {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 40]
    }
    mat_path = 'chronic_stroke/'
    pca_path = 'chronic_stroke/pca_data/'
    save_path = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/manifold_nperseg/'
    trial_min = 13

def load_voxel_data(load_path, subj, roi, trial_min, paradigm, train_stage):
    """Load voxel data for given parameters"""
    data_voxel_list = []
    suffix = '_r' if roi % 2 != 0 else '_l'
    
    logging.info(f"Loading voxel data: subject={subj}, ROI={roi}")
    for num in range(1, trial_min + 1):
        mat_path = os.path.join(
            load_path, subj, str(roi),
            f'{subj}_{paradigm}_{train_stage}_voxel_{num}{suffix}.mat'
        )
        try:
            data_voxel = loadmat(mat_path)['momint_1']
            data_voxel_list.append(data_voxel[:, :200])
        except Exception as e:
            logging.error(f"Failed to load file: {mat_path}")
            raise

    return data_voxel_list

def calculate_psd(data_voxel_list, fs, nperseg, noverlap, freq_band, trial_min):
    """Calculate power spectral density"""
    voxel_psd_list = []
    for trial_ii in range(trial_min):
        f, t, Zxx = signal.stft(
            data_voxel_list[trial_ii], 
            fs, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            scaling='psd'
        )
        freq_mask = np.logical_and(f >= freq_band[0], f <= freq_band[1])
        voxel_psd = np.mean(Zxx[:, freq_mask, :].real, axis=1)
        voxel_psd_ = smooth_average(voxel_psd, 3, 3)[:, 10:40]
        voxel_psd_list.append(voxel_psd_)
    
    return voxel_psd_list

def process_subject_data(config, train_stage, nperseg, roi, subj, freqb):
    """Process data for a single subject"""
    load_path = os.path.join(config.mat_path, train_stage, config.Paradigm)
    noverlap = nperseg - 1
    
    # Load PCA data
    pca_path = os.path.join(
        config.pca_path, train_stage, config.Paradigm,
        subj, str(roi), f'{subj}_pca_trial_{freqb}.npy'
    )
    data_pca = np.load(pca_path)
    
    # Load and process voxel data
    data_voxel_list = load_voxel_data(
        load_path, subj, roi, config.trial_min,
        config.Paradigm, train_stage
    )
    
    # Calculate PSD
    voxel_psd_list = calculate_psd(
        data_voxel_list, config.fs, nperseg, noverlap,
        config.freqBand[freqb], config.trial_min
    )
    
    # Perform PCA and CCA analysis
    psd_pca, _, _ = get_data_mat(voxel_psd_list, 20)
    rank_min = min(
        min(np.linalg.matrix_rank(psd_pca)),
        min(np.linalg.matrix_rank(data_pca))
    )
    
    psd_pca_reshape = np.reshape(np.array(psd_pca)[:, :, :rank_min], (-1, rank_min))
    data_pca_reshape = np.reshape(data_pca[:config.trial_min, :, :rank_min], (-1, rank_min))
    
    r = canoncorr(data_pca_reshape, psd_pca_reshape, fullReturn=False)
    return np.mean(r[:4])

def plot_results(config, data):
    """Plot analysis results"""
    data_ = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], -1))
    
    for stage_num in range(2):
        fig, ax = plt.subplots(ncols=1)
        data_stage = data_[:, stage_num, :, :]
        
        for i in range(data_stage.shape[0]):
            Y = data_stage[i, :, :]
            shaded_errorbar(ax, np.arange(25, 101, 5), Y, label=config.freqb_list[i])
        
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Length of Sliding Window', fontdict={'size': 15})
        ax.set_ylabel('Canonical Correlation', fontdict={'size': 15})
        ax.set_xticks(np.arange(25, 101, 5))
        ax.set_title(f"{config.Paradigm}-{config.train_stage[stage_num]}", fontdict={'size': 15})
        ax.set_ylim([0, 1])
        
        fig.tight_layout()
        plt.show()
        
        save_path = os.path.join(
            config.save_path,
            f'ManiPSDWindowLength_{config.Paradigm}_{config.train_stage[stage_num]}.png'
        )
        fig.savefig(save_path, format='png', dpi=1000)
        logging.info(f"Saved plot to: {save_path}")

def main():
    """Main execution flow"""
    setup_logging()
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    
    logging.info("Starting analysis...")
    CCA_score = []
    
    for freqb in config.freqb_list:
        logging.info(f"Processing frequency band: {freqb}")
        CCA_score_stage = []
        
        for train_stage in config.train_stage:
            logging.info(f"Processing stage: {train_stage}")
            CCA_score_nper = []
            
            for nperseg in config.nperseg_group:
                logging.info(f"Processing window length: {nperseg}")
                CCA_score_roi = []
                
                for roi in config.ROIs:
                    CCA_score_subj = []
                    for subj in config.subj_list:
                        score = process_subject_data(
                            config, train_stage, nperseg, roi, subj, freqb
                        )
                        CCA_score_subj.append(score)
                    CCA_score_roi.append(CCA_score_subj)
                CCA_score_nper.append(CCA_score_roi)
            CCA_score_stage.append(CCA_score_nper)
        CCA_score.append(CCA_score_stage)
    
    # Save results
    save_path = os.path.join(config.save_path, 'ManiPSDNperseg.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(CCA_score, file)
    logging.info(f"Saved results to: {save_path}")
    
    # Plot results
    plot_results(config, np.array(CCA_score))

if __name__ == "__main__":
    main()
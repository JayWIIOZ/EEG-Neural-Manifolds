import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from utils import *

'''
Next step: Investigate if changes in PSD (Power Spectral Density) are reflected in neural manifolds.
'''

def cosine_similarity(x, y):
    """Compute the cosine similarity between two vectors."""
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

class Config:
    """Configuration class for setting up parameters."""
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]  # Regions of Interest
    train_stage = ['pre', 'post']  # Training stages
    Paradigm = 'AO1'  # Experimental paradigm
    freqb = 'alpha'  # Frequency band of interest
    subj_list = ['kmt', 'ock']  # Subject list
    fs = 100  # Sampling frequency
    nperseg = 25  # Segment length for STFT
    noverlap = 24  # Overlap length for STFT
    freqBand = {'delta': [1, 4],
                'theta': [4, 8],
                'alpha': [8, 12],
                'beta': [12, 30],
                'gamma': [30, 40]}  # Frequency bands
    mat_path = 'chronic_stroke/'  # Path to .mat files
    pca_path = 'chronic_stroke/pca_data/'  # Path to PCA data
    save_path = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/manifold_psd/'  # Save path for results
    trial_min = 13  # Minimum number of trials

# Create the save directory if it doesn't exist
os.makedirs(Config.save_path, exist_ok=True)

def main():
    CCA_score = []  # List to store Canonical Correlation Analysis (CCA) scores

    for trainStage in Config.train_stage:
        # Load data for each training stage
        CCA_score_stage = []
        load_path = os.path.join(Config.mat_path, trainStage, Config.Paradigm)

        for roi in Config.ROIs:
            CCA_score_roi = []

            for subj in Config.subj_list:
                data_voxel_list = []
                CCA_score_subj = []
                pca_path = os.path.join(Config.pca_path, trainStage, Config.Paradigm, subj, str(roi), f'{subj}_pca_trial_{Config.freqb}.npy')
                data_pca = np.load(pca_path)

                # Load voxel data based on ROI
                if roi % 2 == 0:
                    for num in range(1, Config.trial_min + 1):
                        mat_path = os.path.join(load_path, subj, str(roi), f'{subj}_{Config.Paradigm}_{trainStage}_voxel_{num}_l.mat')
                        data_voxel = loadmat(mat_path)['momint_1']
                        data_voxel_list.append(data_voxel[:, :200])
                else:
                    for num in range(1, Config.trial_min + 1):
                        mat_path = os.path.join(load_path, subj, str(roi), f'{subj}_{Config.Paradigm}_{trainStage}_voxel_{num}_r.mat')
                        data_voxel = loadmat(mat_path)['momint_1']
                        data_voxel_list.append(data_voxel[:, :200])

                # Compute PSD for each trial
                voxel_psd_list = []
                for trial_ii in range(Config.trial_min):
                    f, t, Zxx = signal.stft(data_voxel_list[trial_ii], Config.fs, nperseg=Config.nperseg, noverlap=Config.noverlap, scaling='psd')
                    voxel_psd = np.mean(Zxx[:, np.logical_and(f >= Config.freqBand[Config.freqb][0], f <= Config.freqBand[Config.freqb][1]), :].real, axis=1)
                    voxel_psd_ = smooth_average(voxel_psd, 3, 3)[:, 10:40]
                    voxel_psd_list.append(voxel_psd_)

                # Perform PCA on PSD data
                psd_pca, var_ratio, _ = get_data_mat(voxel_psd_list, 20)
                rank_min = min(min(np.linalg.matrix_rank(psd_pca)), min(np.linalg.matrix_rank(data_pca)))
                psd_pca_reshape = np.reshape(np.array(psd_pca)[:, :, :rank_min], (-1, rank_min))
                data_pca_reshape = np.reshape(data_pca[:Config.trial_min, :, :rank_min], (-1, rank_min))

                # Compute CCA between PCA of PSD and neural manifold data
                r = canoncorr(data_pca_reshape, psd_pca_reshape, fullReturn=False)
                CCA_score_subj.append(r)
                CCA_score_roi.append(CCA_score_subj)

            # Append CCA scores for each ROI
            dim_min = min([CCA_score_roi[i][0].shape for i in range(len(CCA_score_roi))])[0]
            CCA_score_stage.append([temp[0][:dim_min] for temp in CCA_score_roi])
        CCA_score.append(CCA_score_stage)

    # Visualization of CCA results
    ROIs_label = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']

    # Plot for Pre-training stage
    fig1, ax1 = plt.subplots(ncols=1)
    for i in range(len(CCA_score[0])):
        Y = np.array(CCA_score[0][i])
        ax1.plot(np.arange(1, np.mean(Y, axis=0).shape[0] + 1), np.mean(Y, axis=0), label=str(ROIs_label[i]))
    ax1.legend(fontsize=10)
    ax1.set_xlabel('Dimensions', fontdict={'size': 15})
    ax1.set_ylabel('Canonical Correlation', fontdict={'size': 15})
    ax1.set_title(Config.Paradigm + '-' + Config.freqb + '-Pre', fontdict={'size': 15})
    ax1.set_ylim([0, 1.01])
    ax1.set_xticks(np.arange(0, 19, 2))
    fig1.tight_layout()
    plt.show()
    fig1.savefig(Config.save_path + 'ManiPsd_' + Config.Paradigm + '_' + Config.freqb + '_Pre.png', format='png', dpi=1000)

    # Plot for Post-training stage
    fig2, ax2 = plt.subplots(ncols=1)
    for i in range(len(CCA_score[1])):
        Y = np.array(CCA_score[1][i])
        ax2.plot(np.arange(1, np.mean(Y, axis=0).shape[0] + 1), np.mean(Y, axis=0), label=str(ROIs_label[i]))
    ax2.legend(fontsize=10)
    ax2.set_xlabel('Dimensions', fontdict={'size': 15})
    ax2.set_ylabel('Canonical Correlation', fontdict={'size': 15})
    ax2.set_title(Config.Paradigm + '-' + Config.freqb + '-Post', fontdict={'size': 15})
    ax2.set_ylim([0, 1.01])
    ax2.set_xticks(np.arange(0, 19, 2))
    fig2.tight_layout()
    plt.show()
    fig2.savefig(Config.save_path + 'ManiPsd_' + Config.Paradigm + '_' + Config.freqb + '_Post.png', format='png', dpi=1000)

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import distance
from utils import canoncorr, svd

# Constants
ROIS = [1, 2, 19, 20, 59, 60, 61, 62]
ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
TRAIN_STAGES = ['pre', 'post']
PARADIGM = 'AO1'
FREQ_BAND = 'beta'
SUBJ_LIST = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wwf', 'wsc']
LOAD_PATH = './RESULTS/Multimodality/'

def load_subject_data(subj: str, roi: int, stage: str) -> np.ndarray:
    """Load and preprocess trial data for a subject"""
    data_path = os.path.join(LOAD_PATH, stage, PARADIGM, subj, 'trial', str(roi))
    return np.load(f"{data_path}/{subj}_{PARADIGM}_{stage}_pca_trial_{FREQ_BAND}.npy")

def preprocess_trial_data(data_pre: np.ndarray, data_post: np.ndarray) -> tuple:
    """Preprocess pre/post trial data to same dimensions"""
    trial_min = min(data_pre.shape[0], data_post.shape[0])
    rank = min(np.linalg.matrix_rank(data_pre), np.linalg.matrix_rank(data_post))
    
    data_pre = data_pre[:trial_min, :, :rank]
    data_post = data_post[:trial_min, :, :rank]
    
    return (data_pre.reshape(-1, rank), 
            data_post.reshape(-1, rank),
            data_pre.shape[1])  # time_len

def compute_manifold_distance(data_pre: np.ndarray, 
                            data_post: np.ndarray, 
                            time_len: int) -> float:
    """Compute manifold distance between pre/post data"""
    A1, B1, r1, *_ = canoncorr(data_pre, data_post, fullReturn=True)
    U1, s1, Vh1 = svd(A1, full_matrices=False)
    U2, s2, Vh2 = svd(B1, full_matrices=False)
    
    temp_pre = np.reshape(data_pre @ U1 @ Vh1, (-1, time_len, r1.shape[-1]))
    temp_post = np.reshape(data_post @ U2 @ Vh2, (-1, time_len, r1.shape[-1]))
    
    mani_diff = [distance.euclidean(np.mean(temp_pre,0)[i,:], 
                                  np.mean(temp_post,0)[i,:]) 
                 for i in range(temp_pre.shape[1])]
    return np.mean(mani_diff)

def load_fma_scores() -> np.ndarray:
    """Load and process FMA scores"""
    df = pd.read_excel('./subj_info.xlsx')
    fma_pre = df[df['name'].isin(SUBJ_LIST)]['FMA_Pre'].values
    fma_post = df[df['name'].isin(SUBJ_LIST)]['FMA_Post'].values
    return fma_post - fma_pre

def plot_correlations(correlations: np.ndarray, p_values: np.ndarray):
    """Plot correlation results"""
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(ROIS)), correlations)
    sig_mask = p_values < 0.05
    ax.scatter(np.arange(len(ROIS))[sig_mask], 
              sig_mask.astype(float), 
              marker='*', c='r')
    
    ax.set_xticks(np.arange(len(ROIS)))
    ax.set_xticklabels(ROI_LABELS)
    ax.set_xlabel('Regions of Interest', fontsize=15)
    ax.set_ylabel('Pearson Correlations', fontsize=15)
    ax.set_title(f'{PARADIGM}-{FREQ_BAND}-Correlations', fontsize=15)
    
    plt.tight_layout()
    plt.show()

def main():
    # Process each ROI
    distances = []
    for roi in ROIS:
        subj_distances = []
        for subj in SUBJ_LIST:
            # Load and process data
            data_pre = load_subject_data(subj, roi, TRAIN_STAGES[0])
            data_post = load_subject_data(subj, roi, TRAIN_STAGES[1])
            
            data_pre_proc, data_post_proc, time_len = preprocess_trial_data(data_pre, data_post)
            dist = compute_manifold_distance(data_pre_proc, data_post_proc, time_len)
            subj_distances.append(dist)
            
        distances.append(subj_distances)
    
    # Compute correlations with FMA
    fma_diff = load_fma_scores()
    correlations = []
    p_values = []
    
    for dist in distances:
        corr = spearmanr(dist, fma_diff)
        correlations.append(corr.correlation)
        p_values.append(corr.pvalue)
    
    # Visualize results    
    plot_correlations(np.array(correlations), np.array(p_values))

if __name__ == "__main__":
    main()
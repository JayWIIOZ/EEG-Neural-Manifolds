
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
from utils import canoncorr  # Ensure custom utils are available

# Configuration
HEMISPHERES = [[1,2], [19,20], [59,60], [61,62]]
ROI_LABELS = ['PreCG', 'SMA', 'SPG', 'IPL']
TRAIN_STAGES = ['pre', 'post']
PARADIGM = 'AO1'  # Change to 'rest' for rest paradigm analysis
FREQ_BAND = 'beta'
SUBJECTS = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wsc', 'wwf']

# Path configuration (Update these paths before running)
BASE_PATH = 'G:/CUHK_Intern/RESULTS/Multimodality/'
FMA_PATH = 'G:/CUHK_Intern/subj_info.xlsx'
OUTPUT_DIR = 'F:/CUHK_Intern/RESULTS/figure/Multimodality/'

def load_fma_scores():
    """Load FMA scores from Excel file."""
    df = pd.read_excel(FMA_PATH)
    return {
        'pre': df['FMA_Pre'].values,
        'post': df['FMA_Post'].values,
        'subjects': df['name'].tolist()
    }

def process_hemisphere_data(roi_num):
    """Process data for a single hemisphere ROI."""
    cca_scores = []
    for stage in TRAIN_STAGES:
        stage_data_r, stage_data_l = [], []
        load_path = f"{BASE_PATH}{stage}/{PARADIGM}/"
        
        for subj in SUBJECTS:
            # Load right hemisphere data
            data_r = np.load(f"{load_path}{subj}/trial/{HEMISPHERES[roi_num][0]}/"
                            f"{subj}_{PARADIGM}_{stage}_pca_trial_{FREQ_BAND}.npy")
            # Load left hemisphere data
            data_l = np.load(f"{load_path}{subj}/trial/{HEMISPHERES[roi_num][1]}/"
                            f"{subj}_{PARADIGM}_{stage}_pca_trial_{FREQ_BAND}.npy")
            
            # Align trial dimensions
            trial_min = min(data_r.shape[0], data_l.shape[0])
            rank = min(np.linalg.matrix_rank(data_r), np.linalg.matrix_rank(data_l))
            data_r = data_r[:trial_min, :, :rank]
            data_l = data_l[:trial_min, :, :rank]
            
            # Reshape for CCA
            stage_data_r.append(data_r.reshape(-1, data_r.shape[-1]))
            stage_data_l.append(data_l.reshape(-1, data_l.shape[-1]))
        
        # Align feature dimensions
        min_rank = min(min(d.shape[-1] for d in stage_data_r),
                      min(d.shape[-1] for d in stage_data_l))
        stage_data_r = [d[:, :min_rank] for d in stage_data_r]
        stage_data_l = [d[:, :min_rank] for d in stage_data_l]
        
        # Calculate CCA scores
        cca_stage = [
            canoncorr(r[:, :min_rank], l[:, :min_rank], fullReturn=False)
            for r, l in zip(stage_data_r, stage_data_l)
        ]
        cca_scores.append(cca_stage)
    return cca_scores

def visualize_results(cca_scores, fma_diff):
    """Generate visualization plots."""
    # Visualization 1: CCA Differences
    plt.figure(figsize=(10, 6))
    for roi in range(len(HEMISPHERES)):
        diff = np.array(cca_scores[roi][1]) - np.array(cca_scores[roi][0])
        _, p_val = stats.wilcoxon(np.ravel(cca_scores[roi][1]), np.ravel(cca_scores[roi][0]))
        label = f"{ROI_LABELS[roi]} {'*' if p_val < 0.05 else ''}"
        plt.plot(np.mean(diff, axis=0), label=label)
    
    plt.ylim(-0.15, 0.2)
    plt.title(f"{PARADIGM}-{FREQ_BAND}")
    plt.xlabel("Canonical Components")
    plt.ylabel("CCA Score Difference")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}asymmetry_diff_{PARADIGM}_{FREQ_BAND}.eps", format='eps', dpi=1000)

    # Visualization 2: FMA Correlation
    plt.figure(figsize=(10, 6))
    for roi in range(len(HEMISPHERES)):
        p_values = [
            stats.pearsonr(
                np.array(cca_scores[roi][1])[:,i] - np.array(cca_scores[roi][0])[:,i],
                fma_diff
            ).pvalue
            for i in range(20)
        ]
        plt.plot(range(1,21), p_values, label=ROI_LABELS[roi])
    
    plt.axhline(0.05, color='red', linestyle='--')
    plt.title(f"P-values for FMA Correlation ({PARADIGM}-{FREQ_BAND})")
    plt.xlabel("Canonical Components")
    plt.ylabel("P-value")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}fma_correlation_{PARADIGM}_{FREQ_BAND}.eps", format='eps', dpi=1000)

if __name__ == "__main__":
    # Data processing
    all_cca_scores = [process_hemisphere_data(roi) for roi in range(len(HEMISPHERES))]
    
    # FMA analysis
    fma_data = load_fma_scores()
    fma_diff = fma_data['post'] - fma_data['pre']
    
    # Visualization
    visualize_results(all_cca_scores, fma_diff)
    plt.show()
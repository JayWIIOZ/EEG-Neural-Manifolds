
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
FREQ_BAND = 'alpha'
SUBJECTS = ['kmt', 'ock']
PRE_FMA_SCORE = {'kmt': 13, 'ock': 17}
POST_FMA_SCORE = {'kmt': 16, 'ock': 20}
# Path configuration (Update these paths before running)
BASE_PATH = 'EEG-Neural-Manifolds/dataset/chronic_stroke/pca_data/'
FMA_PATH = 'EEG-Neural-Manifolds/dataset/chronic_stroke/subj_info.xlsx'
OUTPUT_DIR = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/asymmetry_analysis/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# def load_fma_scores():
#     """Load FMA scores from Excel file."""
#     df = pd.read_excel(FMA_PATH)
#     return {
#         'pre': df['FMA_Pre'].values,
#         'post': df['FMA_Post'].values,
#         'subjects': df['name'].tolist()
#     }

def process_hemisphere_data(roi_num):
    """Process data for a single hemisphere ROI with aligned features across stages."""
    cca_scores = []
    # Determine min_rank across both stages to ensure feature alignment
    stage_data_combined_r, stage_data_combined_l = [], []
    
    for stage in TRAIN_STAGES:
        load_path = f"{BASE_PATH}{stage}/{PARADIGM}/"
        for subj in SUBJECTS:
            data_r = np.load(f"{load_path}{subj}/{HEMISPHERES[roi_num][0]}/{subj}_pca_trial_{FREQ_BAND}.npy")
            data_l = np.load(f"{load_path}{subj}/{HEMISPHERES[roi_num][1]}/{subj}_pca_trial_{FREQ_BAND}.npy")
            
            # Align trial dimensions
            trial_min = min(data_r.shape[0], data_l.shape[0])
            data_r_sliced = data_r[:trial_min, :, :]
            data_l_sliced = data_l[:trial_min, :, :]
            
            # Reshape to 2D and store for combined rank calculation
            data_r_2d = data_r_sliced.reshape(-1, data_r_sliced.shape[-1])
            data_l_2d = data_l_sliced.reshape(-1, data_l_sliced.shape[-1])
            stage_data_combined_r.append(data_r_2d)
            stage_data_combined_l.append(data_l_2d)
    
    # Compute global min_rank across all subjects and stages
    min_rank_r = min([np.linalg.matrix_rank(d) for d in stage_data_combined_r])
    min_rank_l = min([np.linalg.matrix_rank(d) for d in stage_data_combined_l])
    min_rank = min(min_rank_r, min_rank_l)
    
    # Process each stage with the global min_rank
    for stage in TRAIN_STAGES:
        stage_data_r, stage_data_l = [], []
        load_path = f"{BASE_PATH}{stage}/{PARADIGM}/"
        for subj in SUBJECTS:
            data_r = np.load(f"{load_path}{subj}/{HEMISPHERES[roi_num][0]}/{subj}_pca_trial_{FREQ_BAND}.npy")
            data_l = np.load(f"{load_path}{subj}/{HEMISPHERES[roi_num][1]}/{subj}_pca_trial_{FREQ_BAND}.npy")
            
            # Align trial dimensions and truncate to min_rank
            trial_min = min(data_r.shape[0], data_l.shape[0])
            data_r = data_r[:trial_min, :, :min_rank]
            data_l = data_l[:trial_min, :, :min_rank]
            
            # Reshape for CCA
            stage_data_r.append(data_r.reshape(-1, min_rank))
            stage_data_l.append(data_l.reshape(-1, min_rank))
        
        # Calculate CCA scores with aligned features
        cca_stage = [
            canoncorr(r, l, fullReturn=False)
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
    plt.savefig(f"{OUTPUT_DIR}asymmetry_diff_{PARADIGM}_{FREQ_BAND}.png", format='png', dpi=1000)

    # Visualization 2: FMA Correlation
    plt.figure(figsize=(10, 6))
    for roi in range(len(HEMISPHERES)):
        
        min_rank = min(np.array(cca_scores[roi][1]).shape[1], np.array(cca_scores[roi][0]).shape[1])
        p_values = [
            stats.pearsonr(
                np.array(cca_scores[roi][1])[:, i] - np.array(cca_scores[roi][0])[:, i],
                fma_diff
            ).pvalue
            for i in range(min_rank)  
        ]
        plt.plot(range(1, min_rank + 1), p_values, label=ROI_LABELS[roi])
    
    plt.axhline(0.05, color='red', linestyle='--')
    plt.title(f"P-values for FMA Correlation ({PARADIGM}-{FREQ_BAND})")
    plt.xlabel("Canonical Components")
    plt.ylabel("P-value")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}fma_correlation_{PARADIGM}_{FREQ_BAND}.png", format='png', dpi=1000)


if __name__ == "__main__":
    # Data processing
    all_cca_scores = [process_hemisphere_data(roi) for roi in range(len(HEMISPHERES))]
    
    # FMA analysis
    fma_diff = PRE_FMA_SCORE['kmt'] - POST_FMA_SCORE['kmt'], PRE_FMA_SCORE['ock'] - POST_FMA_SCORE['ock']
    
    # Visualization
    visualize_results(all_cca_scores, fma_diff)
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy import stats
from utils import canoncorr, divide_pair, shaded_errorbar
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
FREQ_BAND = 'alpha'
PC_NUM = 4
ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
SUBJ_LIST = ['kmt', 'ock']
LOAD_PATH = 'EEG-Neural-Manifolds/dataset/chronic_stroke/'

def load_subject_data(subj, roi, stage):
    """Load and preprocess data for a single subject"""
    data_path = os.path.join(LOAD_PATH, stage, PARADIGM, subj, 'trial', str(roi))
    data = np.load(f"{data_path}/{subj}_{PARADIGM}_{stage}_pca_trial_{FREQ_BAND}.npy")
    return data

def preprocess_data(data_pre, data_post):
    """Preprocess pre/post data to same dimensions"""
    trial_min = min(data_pre.shape[0], data_post.shape[0])
    rank = min(np.linalg.matrix_rank(data_pre), np.linalg.matrix_rank(data_post))
    
    data_pre = data_pre[:trial_min, :, :rank]
    data_post = data_post[:trial_min, :, :rank]
    
    return data_pre.reshape(-1, rank), data_post.reshape(-1, rank)

def compute_cca_scores(data_list_pre, data_list_post):
    """Compute CCA scores between pre and post data"""
    rank_min = min(min(d.shape[1] for d in data_list_pre),
                   min(d.shape[1] for d in data_list_post))
                   
    cca_scores = []
    for pre, post in zip(data_list_pre, data_list_post):
        pre = pre[:,:rank_min]
        post = post[:,:rank_min]
        score = canoncorr(pre, post, fullReturn=False)
        cca_scores.append(score)
        
    return cca_scores

def compute_cross_subject_consistency(data_list):
    """Compute consistency across subjects"""
    time_min = min(data.shape[0] for data in data_list)
    data_list = [data[:time_min,:] for data in data_list]
    
    # Compute aligned scores
    gcca_scores = []
    subj_pairs = divide_pair(data_list)
    for pair in subj_pairs:
        score = canoncorr(data_list[pair[0]], data_list[pair[1]], fullReturn=False)
        gcca_scores.append(score)
        
    # Compute unaligned correlations  
    coef_pairs = []
    for pair in subj_pairs:
        coefs = []
        for dim in range(data_list[pair[0]].shape[-1]):
            r = stats.pearsonr(data_list[pair[0]][:,dim],
                             data_list[pair[1]][:,dim]).statistic
            coefs.append(r)
        coef_pairs.append(np.array(coefs))
        
    return np.array(gcca_scores), np.array(coef_pairs)

def plot_results(gcca_scores, stage_idx):
    """Plot GCCA results"""
    fig, ax = plt.subplots()
    for i, scores in enumerate(gcca_scores):
        shaded_errorbar(ax, np.arange(1,scores.shape[1]+1), 
                       scores.T, label=ROI_LABELS[i])
    
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Neural Modes', fontsize=15)
    ax.set_ylabel('Canonical Correlation', fontsize=15) 
    ax.set_xticks(np.arange(2, 21, 2))
    ax.set_title(f"{PARADIGM}-{FREQ_BAND}-{TRAIN_STAGES[stage_idx]}", 
                fontsize=15)
    ax.set_ylim([0, 0.9])
    
    plt.tight_layout()
    return fig

def main():
    logging.info("Starting analysis...")
    cca_scores = []
    gcca_scores = []
    coef_scores = []
    
    for roi in tqdm(ROIS, desc="Processing ROIs"):
        # Process each ROI
        data_pre_list = []
        data_post_list = []
        logging.info(f"Processing ROI {roi}")
        
        for subj in tqdm(SUBJ_LIST, desc=f"Processing subjects for ROI {roi}"):
            try:
                # Load and preprocess subject data
                logging.info(f"Loading data for subject {subj}")
                data_pre = load_subject_data(subj, roi, TRAIN_STAGES[0])
                data_post = load_subject_data(subj, roi, TRAIN_STAGES[1])
                
                data_pre_proc, data_post_proc = preprocess_data(data_pre, data_post)
                data_pre_list.append(data_pre_proc)
                data_post_list.append(data_post_proc)
                
            except Exception as e:
                logging.error(f"Error processing subject {subj}: {str(e)}")
                continue
        
        try:
            # Compute scores
            logging.info(f"Computing scores for ROI {roi}")
            cca = compute_cca_scores(data_pre_list, data_post_list)
            gcca_pre, coef_pre = compute_cross_subject_consistency(data_pre_list)
            gcca_post, coef_post = compute_cross_subject_consistency(data_post_list)
            
            cca_scores.append(cca)
            gcca_scores.append([gcca_pre, gcca_post])
            coef_scores.append([coef_pre, coef_post])
            
        except Exception as e:
            logging.error(f"Error computing scores for ROI {roi}: {str(e)}")
            continue
    
    # Plot results
    logging.info("Generating plots...")
    for stage in range(2):
        try:
            fig = plot_results([scores[stage] for scores in gcca_scores], stage)
            save_path = f'F:/CUHK_Intern/RESULTS/figure/Multimodality/cross_subj_CCA_{PARADIGM}_{FREQ_BAND}_{TRAIN_STAGES[stage]}.eps'
            fig.savefig(save_path, format='eps', dpi=1000)
            logging.info(f"Plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Error generating plot for stage {stage}: {str(e)}")
    
    logging.info("Analysis completed!")

if __name__ == "__main__":
    main()
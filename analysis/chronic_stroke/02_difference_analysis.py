import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial import distance
from scipy.stats import spearmanr
from utils import canoncorr, divide_pair, shaded_errorbar, svd
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DEFAULT_CONFIG = {
    'ROIS': [1, 2, 19, 20, 59, 60, 61, 62],
    'ROI_LABELS': ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R'],
    'TRAIN_STAGES': ['pre', 'post'],
    'PARADIGM': 'AO1',
    'FREQ_BAND': 'alpha',
    'SUBJ_LIST_CCA': ['kmt', 'ock'],
    'SUBJ_LIST_FMA': ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wwf', 'wsc'],
    'LOAD_PATH_CCA': 'EEG-Neural-Manifolds/dataset/chronic_stroke/',
    'LOAD_PATH_FMA': './RESULTS/Multimodality/',
    'PC_NUM': 4
}

def load_subject_data(subj: str, roi: int, stage: str, mode: str = 'CCA') -> np.ndarray:
    """load and preprocess trial data for a subject"""
    load_path = DEFAULT_CONFIG['LOAD_PATH_CCA'] if mode == 'CCA' else DEFAULT_CONFIG['LOAD_PATH_FMA']
    data_path = os.path.join(load_path, stage, DEFAULT_CONFIG['PARADIGM'], subj, 'trial', str(roi))
    return np.load(f"{data_path}/{subj}_{DEFAULT_CONFIG['PARADIGM']}_{stage}_pca_trial_{DEFAULT_CONFIG['FREQ_BAND']}.npy")

def preprocess_data(data_pre: np.ndarray, data_post: np.ndarray, return_time: bool = False):
    """preprocess pre/post data to same dimensions"""
    trial_min = min(data_pre.shape[0], data_post.shape[0])
    rank = min(np.linalg.matrix_rank(data_pre), np.linalg.matrix_rank(data_post))
    
    data_pre = data_pre[:trial_min, :, :rank]
    data_post = data_post[:trial_min, :, :rank]
    
    if return_time:
        return (data_pre.reshape(-1, rank), 
                data_post.reshape(-1, rank),
                data_pre.shape[1])  # time_len
    return data_pre.reshape(-1, rank), data_post.reshape(-1, rank)

# ================== CCA ==================
def compute_cca_scores(data_list_pre, data_list_post):
    """calculate CCA scores between pre and post data"""
    rank_min = min(min(d.shape[1] for d in data_list_pre),
                   min(d.shape[1] for d in data_list_post))
    return [canoncorr(pre[:,:rank_min], post[:,:rank_min], fullReturn=False) 
            for pre, post in zip(data_list_pre, data_list_post)]

def compute_cross_subject_consistency(data_list):
    """calculate consistency across subjects"""
    time_min = min(data.shape[0] for data in data_list)
    aligned_data = [data[:time_min,:] for data in data_list]
    
    # GCCA scores after alignment
    gcca_scores = [canoncorr(aligned_data[p[0]], aligned_data[p[1]], fullReturn=False)
                  for p in divide_pair(aligned_data)]
    
    # without alignment
    coef_pairs = []
    for pair in divide_pair(aligned_data):
        coefs = [stats.pearsonr(aligned_data[pair[0]][:,dim], 
                               aligned_data[pair[1]][:,dim]).statistic
                for dim in range(aligned_data[pair[0]].shape[-1])]
        coef_pairs.append(np.array(coefs))
    
    return np.array(gcca_scores), np.array(coef_pairs)

def plot_cca_results(gcca_scores, stage_idx):
    """plot CCA results"""
    fig, ax = plt.subplots()
    for i, scores in enumerate(gcca_scores):
        shaded_errorbar(ax, np.arange(1,scores.shape[1]+1), scores.T, 
                       label=DEFAULT_CONFIG['ROI_LABELS'][i])
    
    ax.legend(fontsize=12)
    ax.set(xlabel='Neural Modes', ylabel='Canonical Correlation',
           title=f"{DEFAULT_CONFIG['PARADIGM']}-{DEFAULT_CONFIG['FREQ_BAND']}-{DEFAULT_CONFIG['TRAIN_STAGES'][stage_idx]}",
           xticks=np.arange(2, 21, 2), ylim=[0, 0.9])
    plt.tight_layout()
    return fig

# ================== FMA ==================

def compute_manifold_distance(data_pre, data_post, time_len):
    """calculate manifold distance between pre/post data"""
    A1, B1, r1, *_ = canoncorr(data_pre, data_post, fullReturn=True)
    U1, s1, Vh1 = svd(A1, full_matrices=False)
    U2, s2, Vh2 = svd(B1, full_matrices=False)
    
    temp_pre = np.reshape(data_pre @ U1 @ Vh1, (-1, time_len, r1.shape[-1]))
    temp_post = np.reshape(data_post @ U2 @ Vh2, (-1, time_len, r1.shape[-1]))
    
    return np.mean([distance.euclidean(np.mean(temp_pre,0)[i,:], 
                                     np.mean(temp_post,0)[i,:]) 
                   for i in range(temp_pre.shape[1])])

def load_fma_scores():
    """load and process FMA scores"""
    df = pd.read_excel('./subj_info.xlsx')
    subjects = df[df['name'].isin(DEFAULT_CONFIG['SUBJ_LIST_FMA'])]
    return subjects['FMA_Post'].values - subjects['FMA_Pre'].values

def plot_correlations(correlations, p_values):
    """plot correlation results"""
    fig, ax = plt.subplots()
    ax.bar(range(len(DEFAULT_CONFIG['ROIS'])), correlations)
    
    # mark significant correlations
    sig_mask = p_values < 0.05
    ax.scatter(np.arange(len(DEFAULT_CONFIG['ROIS']))[sig_mask], 
              sig_mask.astype(float), marker='*', c='r')
    
    ax.set_xticks(range(len(DEFAULT_CONFIG['ROIS'])))
    ax.set_xticklabels(DEFAULT_CONFIG['ROI_LABELS'], rotation=45, ha='right')
    ax.set(xlabel='Regions of Interest', ylabel='Spearman Correlation',
           title=f"{DEFAULT_CONFIG['PARADIGM']}-{DEFAULT_CONFIG['FREQ_BAND']}-FMA Correlation")
    plt.tight_layout()
    plt.show()

# ================== Main ==================
def run_cca_analysis():
    """execute CCA analysis"""
    logging.info("Starting CCA analysis...")
    cca_scores, gcca_scores, coef_scores = [], [], []
    
    for roi in tqdm(DEFAULT_CONFIG['ROIS'], desc="Processing ROIs"):
        data_pre_list, data_post_list = [], []
        
        for subj in tqdm(DEFAULT_CONFIG['SUBJ_LIST_CCA'], desc=f"Subjects for ROI {roi}"):
            try:
                data_pre = load_subject_data(subj, roi, DEFAULT_CONFIG['TRAIN_STAGES'][0], mode='CCA')
                data_post = load_subject_data(subj, roi, DEFAULT_CONFIG['TRAIN_STAGES'][1], mode='CCA')
                pre_proc, post_proc = preprocess_data(data_pre, data_post)
                data_pre_list.append(pre_proc)
                data_post_list.append(post_proc)
            except Exception as e:
                logging.error(f"Subject {subj} error: {str(e)}")
                continue
        
        try:
            cca_scores.append(compute_cca_scores(data_pre_list, data_post_list))
            gcca_pre, coef_pre = compute_cross_subject_consistency(data_pre_list)
            gcca_post, coef_post = compute_cross_subject_consistency(data_post_list)
            gcca_scores.append([gcca_pre, gcca_post])
            coef_scores.append([coef_pre, coef_post])
        except Exception as e:
            logging.error(f"ROI {roi} error: {str(e)}")
            continue
    
    # save CCA results
    for stage in range(2):
        try:
            fig = plot_cca_results([scores[stage] for scores in gcca_scores], stage)
            fig.savefig(f'cross_subj_CCA_{DEFAULT_CONFIG["PARADIGM"]}_{DEFAULT_CONFIG["FREQ_BAND"]}_{DEFAULT_CONFIG["TRAIN_STAGES"][stage]}.eps', 
                       format='eps', dpi=1000)
        except Exception as e:
            logging.error(f"Plot error: {str(e)}")

def run_fma_analysis():
    """execute FMA analysis"""
    logging.info("Starting FMA correlation analysis...")
    distances = []
    
    for roi in DEFAULT_CONFIG['ROIS']:
        subj_dists = []
        for subj in DEFAULT_CONFIG['SUBJ_LIST_FMA']:
            try:
                data_pre = load_subject_data(subj, roi, DEFAULT_CONFIG['TRAIN_STAGES'][0], mode='FMA')
                data_post = load_subject_data(subj, roi, DEFAULT_CONFIG['TRAIN_STAGES'][1], mode='FMA')
                pre_proc, post_proc, time_len = preprocess_data(data_pre, data_post, return_time=True)
                subj_dists.append(compute_manifold_distance(pre_proc, post_proc, time_len))
            except Exception as e:
                logging.error(f"Subject {subj} error: {str(e)}")
                continue
        distances.append(subj_dists)
    
    # calculate correlations with FMA
    fma_diff = load_fma_scores()
    corr_results = [spearmanr(dist, fma_diff) for dist in distances]
    plot_correlations([r.correlation for r in corr_results], [r.pvalue for r in corr_results])

if __name__ == "__main__":
    
    run_cca_analysis()   
    # run_fma_analysis()  
"""
AORestCls.py - Classifier for AO and Rest paradigm data analysis

This script loads EEG data from specified ROIs, performs feature extraction using CCA,
trains a classifier (LinearSVC by default), evaluates performance across subjects,
and visualizes results with statistical significance markers.

Usage:
1. Ensure data paths are correctly set in 'CONFIG' section.
2. Adjust parameters in 'CONFIG' as needed (ROIs, classifier, etc.).
3. Run script. Results will be saved as EPS in specified output path.
"""
#%% ---------------------------- CONFIG --------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import LinearSVC
from utils import canoncorr, divide_pair  # Ensure custom utilities are available

# Path configuration
BASE_PATH = 'F:/CUHK_intern/RESULTS/Multimodality/'
OUTPUT_PATH = 'F:/CUHK_Intern/RESULTS/figure/Multimodality/'

# Experiment parameters
ROIS = {
    'ids': [1, 2, 19, 20, 59, 60, 61, 62],
    'labels': ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 
              'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
}
TRAIN_STAGES = ['pre', 'post']
PARADIGMS = ['AO1', 'rest']
FREQ_BAND = 'alpha'
REST_TRIAL_NUM = 13
RANDOM_SEED = 12345

# Classifier configuration
CLASSIFIER = LinearSVC
CLASSIFIER_PARAMS = {'max_iter': 10000}

#%% -------------------------- CORE FUNCTIONS ---------------------------
def load_subject_data(base_path, subj, roi, paradigm, stage, freq_band):
    """Load preprocessed EEG data for a subject.
    
    Args:
        base_path: Root directory for data
        subj: Subject ID
        roi: ROI number
        paradigm: 'AO1' or 'rest'
        stage: 'pre' or 'post'
        freq_band: Frequency band identifier
        
    Returns:
        numpy array: Data matrix of shape (trials, timepoints, components)
    """
    path = f"{base_path}{stage}/{paradigm}/{subj}/trial/{roi}/"
    file_name = f"{subj}_{paradigm}_{stage}_pca_trial_{freq_band}.npy"
    return np.load(path + file_name)

def prepare_dataset(subjects, roi):
    """Prepare combined dataset for all subjects and paradigms.
    
    Args:
        subjects: List of subject IDs
        roi: ROI number
        
    Returns:
        list: Processed data arrays for all subjects
    """
    dataset = []
    for subj in subjects:
        ao_data = load_subject_data(
            BASE_PATH+'pre/', subj, roi, PARADIGMS[0], 'pre', FREQ_BAND)
        rest_data = load_subject_data(
            BASE_PATH+'pre/', subj, roi, PARADIGMS[1], 'pre', FREQ_BAND)[:REST_TRIAL_NUM]
        
        combined = np.vstack([ao_data, rest_data])
        rank = np.linalg.matrix_rank(combined)
        dataset.append(combined[:, :, :rank])
    
    return dataset

def compute_cca_features(temp1, temp2):
    """Compute CCA-transformed features.
    
    Args:
        temp1: Data matrix from subject 1
        temp2: Data matrix from subject 2
        
    Returns:
        tuple: Transformed features (X1_test, X2_test)
    """
    A, B, *_ = canoncorr(temp1, temp2, fullReturn=True)
    X1_test = temp1 @ A @ np.linalg.inv(B)
    X2_test = temp2 @ B @ np.linalg.inv(A)
    return X1_test.reshape((-1, X1_test.shape[1]*X1_test.shape[2])), \
           X2_test.reshape((-1, X2_test.shape[1]*X2_test.shape[2]))

def train_evaluate_classifier(X_train, Y_train, X_test, Y_test):
    """Train and evaluate classifier.
    
    Args:
        X_train: Training features
        Y_train: Training labels
        X_test: Test features
        Y_test: Test labels
        
    Returns:
        float: Classification accuracy
    """
    clf = CLASSIFIER(**CLASSIFIER_PARAMS)
    clf.fit(X_train, Y_train)
    return clf.score(X_test, Y_test)

#%% ------------------------ MAIN PROCESSING ----------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    subjects = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wsc', 'wwf']
    stage_results = []

    for stage in TRAIN_STAGES:
        roi_scores = []
        for roi in ROIS['ids']:
            dataset = prepare_dataset(subjects, roi)
            rank_min = min(d.shape[-1] for d in dataset)
            subject_pairs = divide_pair(dataset)
            
            pair_scores = []
            for pair in subject_pairs:
                subj1_data = dataset[pair[0]]
                subj2_data = dataset[pair[1]]
                trial_min = min(subj1_data.shape[0], subj2_data.shape[0])
                
                # Prepare CCA features
                temp1 = subj1_data[-trial_min:, :, :rank_min].reshape((-1, rank_min))
                temp2 = subj2_data[-trial_min:, :, :rank_min].reshape((-1, rank_min))
                X1, X2 = compute_cca_features(temp1, temp2)
                
                # Classification
                scores = []
                for subj_idx in [0, 1]:
                    X = dataset[pair[subj_idx]][-trial_min:].reshape((trial_min, -1))
                    Y = np.concatenate([np.ones(trial_min-REST_TRIAL_NUM), 
                                      2*np.ones(REST_TRIAL_NUM)])
                    
                    # Shuffle trials
                    idx = rng.permutation(len(Y))
                    X_train, Y_train = X[idx], Y[idx]
                    X_test = X2[idx] if subj_idx == 0 else X1[idx]
                    
                    scores.append(train_evaluate_classifier(X_train, Y_train, X_test, Y[idx]))
                
                pair_scores.append(np.mean(scores))
            roi_scores.append(pair_scores)
        stage_results.append(roi_scores)

    # Statistical testing
    significance = []
    for roi_idx in range(len(ROIS['ids'])):
        _, p_val = stats.wilcoxon(stage_results[0][roi_idx], stage_results[1][roi_idx])
        significance.append(p_val < 0.05)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ROIS['ids'])) + 1
    width = 0.4
    
    for i, (stage, color) in enumerate(zip(['Pre', 'Post'], ['#82B0D2', '#FA7F6F'])):
        means = [np.mean(scores) for scores in stage_results[i]]
        stds = [np.std(scores) for scores in stage_results[i]]
        
        ax.bar(x + (i-0.5)*width, means, width, label=stage, color=color)
        ax.errorbar(x + (i-0.5)*width, means, yerr=stds, fmt='o', 
                   color='red', capsize=5)
    
    # Mark significant ROIs
    sig_x = x[np.array(significance)]
    ax.scatter(sig_x, [1.05]*sum(significance), marker='*', c='r', s=100, zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(ROIS['labels'], rotation=15)
    ax.set_ylim(0.4, 0.75)
    ax.set_xlabel('Regions of Interest', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title(f'{FREQ_BAND.capitalize()} Band Classification Performance', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}AO_Rest_cls_balanceTrial_{FREQ_BAND}.eps", 
               format='eps', dpi=1000)
    print("Processing completed. Results saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
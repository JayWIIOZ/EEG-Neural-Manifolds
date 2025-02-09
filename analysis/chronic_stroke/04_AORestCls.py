import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal, stats
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from utils import *
import logging

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_shuffled_indices(config, size):
    """Get shuffled indices for data"""
    trial_index = np.arange(size)
    # Ensure indices are actually shuffled
    while ((shuffled := config.rng.permutation(trial_index)) == trial_index).all():
        continue
    return shuffled

class Config:
    """Configuration parameters for AO-Rest classification"""
    rng = np.random.default_rng(np.random.SeedSequence(12345))
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
    train_stage = ['pre', 'post']
    Paradigm = ['AO1', 'rest']
    freqb = 'alpha'
    pcNum = 4
    rest_trial = 13
    classifier_model = LinearSVC
    classifier_params = {'max_iter': 10000}
    data_path = 'chronic_stroke/pca_data/'
    save_path = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/classification_results/'
    subj_list = ['kmt', 'ock']
    ROIs_label = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 
                  'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']

def load_data(config, train_stage, roi, subj):
    """Load and prepare data for classification"""
    data_list = []
    for paradigm in config.Paradigm:
        path = os.path.join(
            config.data_path, train_stage, paradigm,
            subj, str(roi), f'{subj}_pca_trial_{config.freqb}.npy'
        )
        try:
            data = np.load(path)
            if paradigm == 'rest':
                data = data[:config.rest_trial, :, :]
            data_list.append(data)
        except Exception as e:
            logging.error(f"Failed to load {path}: {e}")
            raise
    
    return np.vstack(data_list)

def process_classification(config, data_list, rank_min):
    """Process classification for subject pairs"""
    subj_pair = divide_pair(data_list)
    n_time = data_list[0].shape[1]
    pair_scores = []
    
    for pair_num, temp_ind in enumerate(subj_pair):
        logging.info(f"Processing subject pair {pair_num + 1}")
        trial_min = min(data_list[temp_ind[0]].shape[0], 
                       data_list[temp_ind[1]].shape[0])
        
        # Prepare data
        temp1 = data_list[temp_ind[0]][-trial_min:,:,:rank_min].reshape((-1, rank_min))
        temp2 = data_list[temp_ind[1]][-trial_min:,:,:rank_min].reshape((-1, rank_min))
        
        # Canonical correlation analysis
        A, B, *_ = canoncorr(temp1, temp2, fullReturn=True)
        X1_test = temp1 @ A @ np.linalg.inv(B)
        X2_test = temp2 @ B @ np.linalg.inv(A)
        X1_test = X1_test.reshape((-1, n_time * rank_min))
        X2_test = X2_test.reshape((-1, n_time * rank_min))
        
        scores = train_and_evaluate(config, data_list, temp_ind, 
                                  trial_min, rank_min, X1_test, X2_test)
        pair_scores.append(np.mean(scores))
        
    return pair_scores

def train_and_evaluate(config, data_list, temp_ind, trial_min, rank_min, X1_test, X2_test):
    """Train classifier and evaluate performance"""
    scores = []
    for subj_num in range(len(temp_ind)):
        X = data_list[temp_ind[subj_num]][-trial_min:,:,:rank_min].reshape((trial_min,-1))
        Y = np.squeeze(np.hstack([
            np.ones((1, X.shape[0]-config.rest_trial)), 
            2*np.ones((1, config.rest_trial))
        ]))
        
        # Shuffle data
        trial_index = get_shuffled_indices(config, Y.shape[-1])
        X_train, Y_train = X[trial_index, :], Y[trial_index]
        
        # Train classifier
        classifier = config.classifier_model(**config.classifier_params)
        classifier.fit(X_train, Y_train)
        
        # Test
        config.rng.shuffle(trial_index)
        X_test = X2_test if subj_num == 0 else X1_test
        X_test = X_test[trial_index, :]
        Y_test = Y[trial_index]
        
        scores.append(classifier.score(X_test, Y_test))
    return scores

def plot_results(config, y_cls, y_cls_std, sign_diff):
    """Plot classification results"""
    fig, ax = plt.subplots(ncols=1)
    x = np.arange(len(config.ROIs)) + 1
    width = 0.4
    
    # Plot bars and error bars
    for i, (data, std, label, color) in enumerate(zip(
        y_cls, y_cls_std, ['Pre', 'Post'], ['#82B0D2', '#FA7F6F']
    )):
        offset = width/2 * (1 if i else -1)
        rects = ax.bar(x + offset, data, width, label=label, color=color)
        ax.errorbar(x + offset, data, yerr=std, ecolor='red', fmt='.',
                   markerfacecolor=color, markeredgecolor=color, 
                   elinewidth=1.5, capsize=5)
    
    # Add significance markers
    sign = np.ones(len(config.ROIs))
    ax.scatter(np.arange(len(config.ROIs))[sign_diff==1], 
              sign[sign_diff==1], marker='*', c='r')
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(config.ROIs_label, rotation=15)
    ax.set_ylim([0.4, 0.75])
    ax.set_xlabel('Regions of Interest', fontdict={'size': 15})
    ax.set_ylabel('Accuracy', fontdict={'size': 15})
    ax.tick_params(labelsize=12)
    ax.set_title(f"{config.freqb}-Classification Performance", 
                fontdict={'size': 15})
    ax.legend(fontsize=15)
    
    fig.tight_layout()
    plt.show()
    
    # Save figure
    save_path = os.path.join(
        config.save_path, 
        f'AO_Rest_cls_blanceTrial_{config.freqb}.png'
    )
    fig.savefig(save_path, format='png', dpi=1000)
    logging.info(f"Saved plot to {save_path}")

def main():
    """Main execution flow"""
    setup_logging()
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    logging.info("Starting AO-Rest classification analysis")
    
    stage_scores = []
    for train_stage in config.train_stage:
        logging.info(f"Processing {train_stage} stage")
        roi_scores = []
        
        for roi in config.ROIs:
            logging.info(f"Processing ROI {roi}")
            data_list = []
            
            # Load data for each subject
            for subj in config.subj_list:
                data_list_ = load_data(config, train_stage, roi, subj)
                rank = min(np.linalg.matrix_rank(data_list_))
                data_list.append(data_list_[:,:,:rank])
            
            # Process classification
            rank_min = min(data.shape[-1] for data in data_list)
            roi_scores.append(
                process_classification(config, data_list, rank_min)
            )
            
        stage_scores.append(roi_scores)
    
    # Statistical analysis
    sign_diff = np.zeros(len(config.ROIs))
    for roi_num in range(len(config.ROIs)):
        _, p = stats.wilcoxon(
            stage_scores[0][roi_num], 
            stage_scores[1][roi_num]
        )
        sign_diff[roi_num] = 1 if p < 0.05 else 0
    
    # Prepare visualization data
    y_cls = []
    y_cls_std = []
    for stage_data in stage_scores:
        y_cls.append([np.mean(roi_data) for roi_data in stage_data])
        y_cls_std.append([np.std(roi_data) for roi_data in stage_data])
    
    # Plot results
    plot_results(config, np.array(y_cls), np.array(y_cls_std), sign_diff)

if __name__ == '__main__':
    main()
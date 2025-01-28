from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from utils import canoncorr, divide_pair
from config import Config


    
def load_data(subj: str, roi: int, freq_band: str, Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load AO and rest data for a subject and ROI
    
    Args:
        subj: Subject ID
        roi: ROI number
        freq_band: Frequency band name
    
    Returns:
        Tuple containing AO and rest data
    """
    # Load AO data
    ao_path = os.path.join(Config.AO_PATH, subj, f'roi_{roi}')
    ao_file = f"{subj}_{Config.PARADIGMS[0]}_pca_trial_{freq_band}.npy"
    data_ao = np.load(os.path.join(ao_path, ao_file))
    
    # Load rest data
    rest_path = os.path.join(Config.REST_PATH, subj, f'roi_{roi}')
    rest_file = f"{subj}_{Config.PARADIGMS[1]}_pca_trial_{freq_band}.npy"
    data_rest = np.load(os.path.join(rest_path, rest_file))
    
    return data_ao, data_rest

def prepare_classification_data(data_list: List[np.ndarray], Config) -> Tuple[np.ndarray, np.ndarray]:

    trial_min = min(data.shape[0] for data in data_list)
    rank_min = min(data.shape[-1] for data in data_list)
    
    X = np.vstack([data[-trial_min:, :, :rank_min].reshape((trial_min, -1)) for data in data_list])
    Y = np.hstack([np.ones(trial_min - Config.REST_TRIALS), 2*np.ones(Config.REST_TRIALS)])
    
    return X, Y


def train_and_evaluate(X1: np.ndarray, X2: np.ndarray, Y: np.ndarray, 
                      rng: np.random.Generator, Config) -> float:
    """
    Train classifier and evaluate performance
    
    Args:
        X1, X2: Training and test features
        Y: Labels
        rng: Random number generator
        
    Returns:
        Classification accuracy
    """
    classifier = RandomForestClassifier(random_state=Config.RANDOM_SEED)
    
    # Shuffle data
    trial_index = rng.permutation(Y.shape[0])
    X_train, Y_train = X1[trial_index], Y[trial_index]
    
    # Train and evaluate
    classifier.fit(X_train, Y_train)
    return classifier.score(X2[trial_index], Y[trial_index])

def visualize_results(results: np.ndarray, std_errors: np.ndarray, 
                     significant_rois: np.ndarray, freq_band: str, Config):
    """
    Visualize classification results
    
    Args:
        results: Classification accuracies
        std_errors: Standard errors
        significant_rois: Significant ROIs mask
        freq_band: Frequency band name
    """
    fig, ax = plt.subplots(ncols=1)
    x = np.arange(len(Config.ROIS)) + 1
    width = 0.6
    
    # Plot bars
    ax.bar(x, results, width, label='Pre', color='#82B0D2')
    ax.errorbar(x, results, yerr=std_errors, ecolor='red', fmt='.',
                markerfacecolor='#82B0D2', markeredgecolor='#82B0D2', 
                elinewidth=1.5, capsize=5)
    
    # Add significance markers
    ax.scatter(x[significant_rois==1]-1, np.ones(sum(significant_rois)), 
              marker='*', c='r')
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(Config.ROI_LABELS)
    ax.set_ylim([0.4, 0.75])
    ax.set_xlabel('Regions of Interest', fontdict={'size':15})
    ax.set_ylabel('Accuracy', fontdict={'size':15})
    ax.set_title(f'{freq_band} Cross-Subject Classification Performance', 
                 fontdict={'size':15})
    
    fig.tight_layout()
    plt.show()
    
    # Save figure
    save_file = f'AO_Rest_cls_blanceTrial_{freq_band}_2.eps'
    save_path = '/analysis/results/ao_rest_cls'
    fig.savefig(os.path.join(save_path, save_file), format='eps', dpi=1000)

def main():
    """Main execution function"""
    Config = Config()
    rng = np.random.default_rng(Config.RANDOM_SEED)
    subj_list = os.listdir(Config.REST_PATH)
    
    for freq_band in Config.FREQUENCY_BANDS:
        roi_scores = []
        
        for roi in Config.ROIS:
            # Load and process data
            data_list = []
            for subj in subj_list:
                data_ao, data_rest = load_data(subj, roi, freq_band)
                combined_data = np.vstack([data_ao, data_rest])
                data_list.append(combined_data)
            
            # Perform classification
            subj_pairs = divide_pair(data_list)
            pair_scores = []
            
            for pair in subj_pairs:
                X1, Y = prepare_classification_data([data_list[i] for i in pair], Config)
                X2, _ = prepare_classification_data([data_list[i] for i in reversed(pair)], Config)
                score = train_and_evaluate(X1, X2, Y, rng, Config)
                pair_scores.append(score)
                
            roi_scores.append(pair_scores)
        
        # Statistical analysis
        mean_scores = np.mean(roi_scores, axis=1)
        std_scores = np.std(roi_scores, axis=1)
        significant = stats.wilcoxon(roi_scores[0], roi_scores[1])[1] < 0.05
        
        # Visualize results
        visualize_results(mean_scores, std_scores, significant, freq_band, Config)

if __name__ == "__main__":
    main()
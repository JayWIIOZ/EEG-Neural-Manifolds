import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from sklearn.decomposition import PCA
import os
from utils import *
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
ROIs_LABEL = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
PARADIGM = 'AO1'
FREQ_BAND = 'beta'
TRAIN_STAGE = 'pre'
THRESHOLD = 1  # Range: 0-1
# TRIAL_NUM = 26  # for resting state
TIME_POINTS = 200
SAMPLE_RATE = 100

# Path configuration
BASE_PATH = 'EEG-Neural-Manifolds/dataset/chronic_stroke/'
SUBJECTS = ['kmt', 'ock']
SAVE_PATH = 'EEG-Neural-Manifolds/analysis/results/chronic_stroke/PCA_variance/'
os.makedirs(SAVE_PATH, exist_ok=True)

def load_and_process_data(subject, roi):
    """Load and process EEG data for a given subject and ROI."""
    logging.info(f"Processing subject {subject}, ROI {roi}")
    mom_voxel_list = []
    hemisphere = 'l' if roi % 2 == 0 else 'r'
    
    # Determine the number of trials by counting the files
    trial_path = os.path.join(BASE_PATH, subject)
    trial_num = len([name for name in os.listdir(trial_path) if os.path.isfile(os.path.join(trial_path, name)) and name.endswith('.mat')])
    
    for num in tqdm(range(1, trial_num + 1), desc=f"Loading trials for {subject}"):
        try:
            file_path = f'{BASE_PATH}{subject}/{roi}/{subject}_{PARADIGM}_{TRAIN_STAGE}_voxel_{num}_{hemisphere}.mat'
            mom_voxel = loadmat(file_path)['momint_1']
            
            # Apply bandpass filter
            data_filter = eeg_bp_filter(mom_voxel[:, :TIME_POINTS], fs=SAMPLE_RATE, freqb=FREQ_BAND)
            mom_voxel_list.append(data_filter)
        except Exception as e:
            logging.error(f"Error processing trial {num}: {str(e)}")
            continue
    
    return mom_voxel_list

def apply_threshold_and_smooth(mom_voxel_list, threshold):
    """Apply threshold and smoothing to the data."""
    mom_temp = np.concatenate(mom_voxel_list, 1)
    
    # Find appropriate threshold
    for thres in range(int(np.mean(np.abs(mom_temp), 1).min()), int(np.mean(np.abs(mom_temp), 1).max())):
        voxels_idx = np.mean(np.abs(mom_temp), 1) >= thres
        if np.sum(voxels_idx) / mom_temp.shape[0] <= threshold:
            return [smooth_average(data[voxels_idx, :], 3, 3) for data in mom_voxel_list]
    
    return mom_voxel_list

def visualize_variance(variance_data, save_path):
    """Create and save variance plot."""
    fig, ax = plt.subplots(ncols=1)
    
    for i, var in enumerate(variance_data):
        var_temp = np.reshape(np.array(var), (-1, np.array(var).shape[-1]))
        shaded_errorbar(ax, np.arange(1, 21), var_temp.T, label=ROIs_LABEL[i])
    
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlabel('Principal Components', fontdict={'size': 15})
    ax.set_ylabel('Sum of Explained Variances', fontdict={'size': 15})
    ax.set_title(f'{PARADIGM}-{FREQ_BAND}-{TRAIN_STAGE}', fontdict={'size': 15})
    ax.set_xticks(np.arange(2, 21, 2))
    
    fig.tight_layout()
    save_file = f'{save_path}{PARADIGM}_{TRAIN_STAGE}_{FREQ_BAND}_var.eps'
    fig.savefig(save_file, format='eps', dpi=1000)
    plt.show()

def main():
    logging.info("Starting PCA variance analysis...")
    # Calculate variance for each ROI and subject
    variance_results = []
    for roi in tqdm(ROIs, desc="Processing ROIs"):
        roi_variance = []
        for subject in tqdm(SUBJECTS, desc=f"Processing subjects for ROI {roi}"):
            try:
                # Load and process data
                mom_voxel_list = load_and_process_data(subject, roi)
                logging.info(f"Applying threshold and smoothing for {subject}")
                
                # Apply threshold and smoothing
                mom_avg_list = apply_threshold_and_smooth(mom_voxel_list, THRESHOLD)
                
                # Apply Gaussian smoothing
                win = norm_gauss_window(0.03, 0.05)
                mom_smooth_list = [smooth_data(data.T, win=win, backend='convolve1d')[10:40, :].T 
                                 for data in mom_avg_list]
                
                # Calculate PCA variance
                _, var_ratio = get_data_mat(mom_smooth_list, 20)
                roi_variance.append(var_ratio)
                logging.info(f"Completed processing for subject {subject}, ROI {roi}")
            
            except Exception as e:
                logging.error(f"Error processing subject {subject}: {str(e)}")
                continue
            
        variance_results.append(roi_variance)
    
    logging.info("Visualization started...")
    # Visualize results
    visualize_variance(variance_results, SAVE_PATH)
    logging.info("Analysis completed successfully!")

if __name__ == '__main__':
    main()
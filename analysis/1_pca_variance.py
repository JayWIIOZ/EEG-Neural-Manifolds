import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from utils import *

# Configuration constants
CONFIG = {
    'ROIS': [1, 2, 19, 20, 59, 60, 61, 62],
    'PARADIGM': 'AO',
    'FREQ_BAND': ['alpha', 'beta', 'theta', 'delta'],
    'THRESHOLD': 1,  # Range: 0-1
    'ROI_LABELS': ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R'],
    'DATA_PATH': '/dataset/acute_storke/RESULTS/data',
    'SAVE_PATH': '/dataset/acute_storke/RESULTS/voxel_npy'
}


def load_and_filter_data(subj, roi, trial_num, load_path,freqb):
    """
    Load and filter voxel data for a specific subject and ROI.
    
    Args:
        subj (str): Subject identifier
        roi (int): ROI number
        trial_num (int): Number of trials
        load_path (str): Path to load data from
        
    Returns:
        list: List of filtered data
    """
    mom_voxel_list = []
    side = 'l' if roi % 2 == 0 else 'r'
    
    for num in range(1, trial_num + 1):
        file_path = os.path.join(load_path, subj, f'{subj}_trial_roi_{roi}', 
                               subj, 'trial', str(roi),
                               f'{subj}_voxel_{num}_{side}.mat')
        
        try:
            mom_voxel = sio.loadmat(file_path)['momint_1']
            data_filter = eeg_bp_filter(mom_voxel[:, :200], fs=100, freqb=freqb)
            mom_voxel_list.append(data_filter)
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            
    return mom_voxel_list


def process_subject_data(mom_voxel_list, threshold):
    """
    Process voxel data with thresholding and smoothing.
    
    Args:
        mom_voxel_list (list): List of voxel data
        threshold (float): Threshold value
        
    Returns:
        tuple: Processed PCA data and variance ratio
    """
    mom_temp = np.concatenate(mom_voxel_list, 1)
    
    # Thresholding
    for thres in range(int(np.mean(np.abs(mom_temp), 1).min()), 
                      int(np.mean(np.abs(mom_temp), 1).max())):
        voxels_idx = np.mean(np.abs(mom_temp), 1) >= thres
        percent = np.sum(voxels_idx) / mom_temp.shape[0]
        
        if percent <= threshold:
            mom_avg_list = [smooth_average(mv[voxels_idx, :], 3, 3) 
                          for mv in mom_voxel_list]
            break
    
    # Smoothing
    win = norm_gauss_window(0.03, 0.05)
    mom_smooth_list = [smooth_data(ma.T, win=win, backend='convolve1d')[10:40, :].T 
                      for ma in mom_avg_list]
    
    return get_data_mat(mom_smooth_list, 30)


def visualize_variance(variance_data, save_path, freqb):
    """
    Visualize PCA variance results.
    
    Args:
        variance_data (list): List of variance data for each ROI
        save_path (str): Path to save the figure
    """
    fig, ax = plt.subplots(ncols=1)
    
    for i, var in enumerate(variance_data):
        var_temp = np.reshape(np.array(var), (-1, np.array(var).shape[-1]))
        shaded_errorbar(ax, np.arange(1, 31), var_temp.T, label=CONFIG['ROI_LABELS'][i])
    
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlabel('Principal Components', fontdict={'size': 15})
    ax.set_ylabel('Sum of Explained Variances', fontdict={'size': 15})
    ax.set_title(f"{CONFIG['PARADIGM']}-{freqb}", fontdict={'size': 15})
    ax.set_xticks(np.arange(2, 31, 2))
    fig.tight_layout()
    
    save_file = os.path.join(save_path, f"{CONFIG['PARADIGM']}_{freqb}_var.eps")
    fig.savefig(save_file, format='eps', dpi=1000)
    plt.show()
    
    
def main():
    
    subj_list = sorted(os.listdir(CONFIG['DATA_PATH']))
    for freq_band in CONFIG['FREQ_BAND']:
        variance_data = []
        
        for roi in CONFIG['ROIS']:
            roi_var = []
            
            for subj in subj_list:
                # Count trials
                trial_num = sum(1 for f in os.listdir(os.path.join(CONFIG['DATA_PATH'], subj)) 
                            if f.endswith('.mat'))
                
                # Process data
                mom_voxel_list = load_and_filter_data(subj, roi, trial_num, CONFIG['DATA_PATH'],freq_band)
                data_pca, var_ratio = process_subject_data(mom_voxel_list, CONFIG['THRESHOLD'])
                
                # Save results
                save_dir = os.path.join(CONFIG['SAVE_PATH'], subj, f'roi_{roi}')
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, 
                    f"{subj}_{CONFIG['PARADIGM']}_pca_trial_{freq_band}.npy"), 
                    data_pca)
                
                roi_var.append(var_ratio)
            
            variance_data.append(roi_var)
        
        # Visualize results
        visualize_variance(variance_data, CONFIG['SAVE_PATH'], freq_band)

if __name__ == "__main__":
    main()
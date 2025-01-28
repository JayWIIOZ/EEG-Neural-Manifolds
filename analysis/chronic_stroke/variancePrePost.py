import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from sklearn.decomposition import PCA
from scipy import stats
from utils import *

# Configuration
CONFIG = {
    'rois': [1, 2, 19, 20, 59, 60, 61, 62],
    'roi_labels': ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R'],
    'paradigm': 'rest',
    'freq_band': 'beta',
    'threshold': 1,
    'trial_num': 26,
    'time_points': 200,
    'sample_rate': 100,
    'pc_num': 20,
    'subjects': ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wsc', 'wwf'],
    'stages': ['pre', 'post']
}

# Path configuration
PATHS = {
    'base': 'F:/CUHK_intern/RESULTS/Multimodality',
    'figures': 'F:/CUHK_intern/RESULTS/figure/Multimodality'
}

def load_eeg_data(subject, roi, stage, trial_num):
    """Load EEG data for given parameters."""
    hemisphere = 'l' if roi % 2 != 0 else 'r'
    mom_voxel_list = []
    
    for num in range(1, trial_num + 1):
        file_path = f"{PATHS['base']}/{stage}/{CONFIG['paradigm']}/{subject}/trial/{roi}/{subject}_{CONFIG['paradigm']}_{stage}_voxel_{num}_{hemisphere}.mat"
        mom_voxel = loadmat(file_path)['momint_1']
        data_filter = eeg_bp_filter(mom_voxel[:, :CONFIG['time_points']], 
                                  fs=CONFIG['sample_rate'], 
                                  freqb=CONFIG['freq_band'])
        mom_voxel_list.append(data_filter)
    
    return mom_voxel_list

def process_data(mom_voxel_list):
    """Process data with threshold and smoothing."""
    mom_temp = np.concatenate(mom_voxel_list, 1)
    thresholds = range(int(np.mean(np.abs(mom_temp), 1).min()),
                      int(np.mean(np.abs(mom_temp), 1).max()))
    
    for thres in thresholds:
        voxels_idx = np.mean(np.abs(mom_temp), 1) >= thres
        if np.sum(voxels_idx) / mom_temp.shape[0] <= CONFIG['threshold']:
            mom_avg_list = [smooth_average(data[voxels_idx, :], 3, 3) 
                          for data in mom_voxel_list]
            
            win = norm_gauss_window(0.03, 0.05)
            return [smooth_data(data.T, win=win, backend='convolve1d')[10:40, :].T 
                   for data in mom_avg_list]
    
    return mom_voxel_list

def calculate_pca_variance(data_list, n_components=20):
    """Calculate PCA variance ratios."""
    model = PCA(n_components=n_components, svd_solver='full')
    rates = np.concatenate(data_list, axis=1)
    rates_model = model.fit(rates.T)
    return rates_model.explained_variance_ratio_

def analyze_stage_differences():
    """Analyze PCA variance differences between stages."""
    variance_stages = []
    
    for stage in CONFIG['stages']:
        variance_rois = []
        for roi in CONFIG['rois']:
            roi_variance = []
            for subject in CONFIG['subjects']:
                # Load and process data
                raw_data = load_eeg_data(subject, roi, stage, CONFIG['trial_num'])
                processed_data = process_data(raw_data)
                variance = calculate_pca_variance(processed_data, CONFIG['pc_num'])
                roi_variance.append(variance)
            variance_rois.append(roi_variance)
        variance_stages.append(variance_rois)
    
    return np.array(variance_stages)

def plot_variance_differences(variance_data):
    """Plot variance differences between stages."""
    variance_diff = variance_data[0,:,:,:] - variance_data[1,:,:,:]
    
    # Calculate statistical significance
    labels = []
    for roi_num in range(variance_data.shape[1]):
        _, p = stats.wilcoxon(np.reshape(variance_data[0, roi_num, :, :], -1),
                             np.reshape(variance_data[1, roi_num, :, :], -1))
        labels.append(f"{CONFIG['roi_labels'][roi_num]}{'*' if p < 0.05 else ''}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(variance_diff.shape[0]):
        ax.plot(np.mean(variance_diff[i,:,:10], axis=0), label=labels[i])
    
    ax.set_xlabel('Principal Components', fontsize=15)
    ax.set_ylabel('Difference of Explained Variances', fontsize=15)
    ax.set_title(f"{CONFIG['paradigm']}-{CONFIG['freq_band']}", fontsize=15)
    ax.set_xticks(np.arange(2, 11, 2))
    ax.tick_params(labelsize=12)
    ax.set_ylim([-0.15, 0.25])
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{PATHS['figures']}/{CONFIG['paradigm']}_{CONFIG['freq_band']}_varDiff.eps", 
                format='eps', dpi=1000)
    plt.show()

def main():
    """Main execution function."""
    variance_data = analyze_stage_differences()
    plot_variance_differences(variance_data)

if __name__ == '__main__':
    main()
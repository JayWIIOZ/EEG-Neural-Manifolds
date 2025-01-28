import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from mat73 import loadmat
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial import distance
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from utils import (eeg_bp_filter, smooth_average, norm_gauss_window,
                  smooth_data, shaded_errorbar, canoncorr, divide_pair)

# ================== CONFIG ==================
class Config:
    
    def __init__(self, mode='PCA_VAR'):
        
        self.ROIS = [1, 2, 19, 20, 59, 60, 61, 62]
        self.ROI_LABELS = ['PreCG.L', 'PreCG.R', 'SMA.L', 'SMA.R', 
                          'SPG.L', 'SPG.R', 'IPL.L', 'IPL.R']
        self.SUBJECTS =  ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wwf', 'wsc']
        self.PARADIGMS = ['AO1', 'rest']
        self.FREQ_BANDS = ['alpha', 'beta']
        self.THRESHOLD = 1
        self.PC_NUM = 20
        
        self.PATHS = {
            'raw_data': 'EEG-Neural-Manifolds/dataset/chronic_stroke/',
            'processed_data': './RESULTS/Multimodality/',
            'figures': './RESULTS/figure/Multimodality/'
        }
        
        # 模式相关配置
        self.mode = mode
        self._init_mode_config(mode)
        
    def _init_mode_config(self, mode):
        """set para according to mode"""
        if mode == 'PCA_VAR':
            self.time_points = 200
            self.sample_rate = 100
            self.trial_num = 26
        elif mode == 'FMA_CORR':
            self.stages = ['pre', 'post']
        elif mode == 'STAGE_DIFF':
            self.pc_components = 10

# ================== 核心函数 ==================
class EEGAnalyzer:

    def __init__(self, config):
        self.cfg = config
        os.makedirs(self.cfg.PATHS['figures'], exist_ok=True)
        
    def load_raw_data(self, subject, roi, stage):
        
        hemisphere = 'l' if roi % 2 == 0 else 'r'
        data_list = []
        
        for trial in range(1, self._get_trial_num(subject, stage)+1):
            path = f"{self.cfg.PATHS['raw_data']}{stage}/{self.cfg.PARADIGM}/{subject}/trial/{roi}/"
            file = f"{subject}_{self.cfg.PARADIGM}_{stage}_voxel_{trial}_{hemisphere}.mat"
            try:
                data = loadmat(os.path.join(path, file))['momint_1']
                data_list.append(
                    eeg_bp_filter(data[:, :self.cfg.time_points], 
                                fs=self.cfg.sample_rate, 
                                freqb=self.cfg.FREQ_BAND)
                )
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
        return data_list

    def preprocess_data(self, raw_data):
        """preprocess"""
        
        merged = np.concatenate(raw_data, axis=1)
        for thres in range(int(np.mean(np.abs(merged), 1).min()),
                          int(np.mean(np.abs(merged), 1).max())):
            voxel_mask = np.mean(np.abs(merged), 1) >= thres
            if np.sum(voxel_mask)/merged.shape[0] <= self.cfg.THRESHOLD:
                break
                
        # smoothing
        smoothed = []
        win = norm_gauss_window(0.03, 0.05)
        for data in raw_data:
            masked = smooth_average(data[voxel_mask, :], 3, 3)
            smoothed.append(
                smooth_data(masked.T, win=win, backend='convolve1d')[10:40, :].T
            )
        return smoothed

    def calculate_pca(self, data_list, n_components):
        
        model = PCA(n_components=n_components, svd_solver='full')
        concat_data = np.concatenate(data_list, axis=1)
        model.fit(concat_data.T)
        return model.explained_variance_ratio_

    def analyze_pca_variance(self):
        """Mode 1: PCA analysis of variance"""
        variance_results = []
        for roi in tqdm(self.cfg.ROIS, desc="Processing ROIs"):
            roi_result = []
            for subject in self.cfg.SUBJECTS['PCA']:
                try:
                    raw = self.load_raw_data(subject, roi, 'pre')
                    processed = self.preprocess_data(raw)
                    var_ratio = self.calculate_pca(processed, self.cfg.PC_NUM)
                    roi_result.append(var_ratio)
                except Exception as e:
                    logging.error(f"Error processing {subject}: {str(e)}")
            variance_results.append(roi_result)
        self._plot_variance(variance_results)

    def analyze_fma_correlation(self):
        """Mode 2: FMA correlation analysis"""
        # load FMA score
        df = pd.read_excel('./subj_info.xlsx')
        fma_diff = df[df['name'].isin(self.cfg.SUBJECTS['FMA'])]['FMA_Post'] - df['FMA_Pre']
        
        # Calculate the variance change for each ROI
        var_changes = []
        for roi in self.cfg.ROIS:
            roi_var = []
            for subject in self.cfg.SUBJECTS['FMA']:
                pre_data = self.preprocess_data(
                    self.load_raw_data(subject, roi, 'pre'))
                post_data = self.preprocess_data(
                    self.load_raw_data(subject, roi, 'post'))
                var_diff = self.calculate_pca(post_data, 20) - self.calculate_pca(pre_data, 20)
                roi_var.append(var_diff[:self.cfg.pc_components])
            var_changes.append(roi_var)
        
        
        correlations = []
        for roi_var in var_changes:
            corr = spearmanr(np.mean(roi_var, axis=1), fma_diff)
            correlations.append(corr.correlation)
        self._plot_correlations(correlations)

    def analyze_stage_difference(self):
        """Mode 3: Stage difference analysis"""
        stage_vars = []
        for stage in ['pre', 'post']:
            stage_data = []
            for roi in self.cfg.ROIS:
                roi_data = [
                    self.calculate_pca(
                        self.preprocess_data(
                            self.load_raw_data(subj, roi, stage)), 20)
                    for subj in self.cfg.SUBJECTS['FMA']
                ]
                stage_data.append(roi_data)
            stage_vars.append(stage_data)
        
        
        diff = np.array(stage_vars[0]) - np.array(stage_vars[1])
        p_values = [
            stats.wilcoxon(diff[i].flatten()).pvalue 
            for i in range(len(self.cfg.ROIS))
        ]
        self._plot_stage_diff(diff, p_values)

    # ================== Visualize ==================
    def _plot_variance(self, data):
        
        fig, ax = plt.subplots(figsize=(10,6))
        for i, roi_data in enumerate(data):
            avg = np.mean(roi_data, axis=0)
            shaded_errorbar(ax, np.arange(1, len(avg)+1), avg, 
                           label=self.cfg.ROI_LABELS[i])
        ax.set(xlabel='Principal Components', ylabel='Explained Variance',
              title=f'{self.cfg.PARADIGM}-{self.cfg.FREQ_BAND}')
        ax.legend()
        fig.savefig(f"{self.cfg.PATHS['figures']}pca_variance.eps", format='eps')

    def _plot_correlations(self, corrs):
        
        fig, ax = plt.subplots()
        ax.bar(self.cfg.ROI_LABELS, corrs)
        ax.set_xticklabels(self.cfg.ROI_LABELS, rotation=45, ha='right')
        ax.set_title('FMA Correlation')
        plt.tight_layout()
        plt.show()

    def _plot_stage_diff(self, diff_data, p_values):
        
        fig, ax = plt.subplots()
        for i in range(len(self.cfg.ROIS)):
            label = f"{self.cfg.ROI_LABELS[i]}{'*' if p_values[i]<0.05 else ''}"
            ax.plot(np.mean(diff_data[i], axis=0)[:10], label=label)
        ax.legend()
        fig.savefig(f"{self.cfg.PATHS['figures']}stage_diff.eps", format='eps')

# ================== MAIN ==================
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

    
    config = Config(mode='PCA_VAR')  # PCA_VAR, FMA_CORR, STAGE_DIFF
    analyzer = EEGAnalyzer(config)
    

    if config.mode == 'PCA_VAR':
        analyzer.analyze_pca_variance()
    elif config.mode == 'FMA_CORR':
        analyzer.analyze_fma_correlation()
    elif config.mode == 'STAGE_DIFF':
        analyzer.analyze_stage_difference()
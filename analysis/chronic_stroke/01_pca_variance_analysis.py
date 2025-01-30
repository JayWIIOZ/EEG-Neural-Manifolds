import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from utils import get_data_mat
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
        self.SUBJECTS =  ['kmt','ock']
        self.PARADIGMS = ['AO1', 'rest']
        self.STAGES = ['pre', 'post']
        self.FREQ_BANDS = ['alpha', 'beta']
        self.THRESHOLD = 1
        self.PC_NUM = 4
        self.TRIAL_NUM = 13
        self.PATHS = {
            'raw_data': 'EEG-Neural-Manifolds/dataset/chronic_stroke/',
            'pca_data': 'EEG-Neural-Manifolds/dataset/chronic_stroke/',
            'figures': 'EEG-Neural-Manifolds/analysis/results/chronic_stroke/PCA_variance/'
        }
        self.time_points = 200
        self.sample_rate = 100
        
        # 模式相关配置
        self.mode = mode
        self._init_mode_config(mode)
        
    def _init_mode_config(self, mode):
        """set para according to mode"""
        if mode == 'PCA_VAR':
            self.time_points = 200
            self.sample_rate = 100
            self.trial_num = 13
        elif mode == 'FMA_CORR':
            self.stages = ['pre', 'post']
        elif mode == 'STAGE_DIFF':
            self.pc_components = 10

# ================== 核心函数 ==================
class EEGAnalyzer:

    def __init__(self, config, stage, paradigm, freq_band):
        self.cfg = config
        os.makedirs(self.cfg.PATHS['figures'], exist_ok=True)
        self.stage = stage
        self.paradigm = paradigm
        self.freq_band = freq_band
        self.paths = self.cfg.PATHS['raw_data']
        self.save_path = self.cfg.PATHS['pca_data']
        self.figure_path = self.cfg.PATHS['figures']
    
    def load_and_process_data(self, subject, roi):
        """Load and process EEG data for a given subject and ROI."""
        logging.info(f"Processing subject {subject}, ROI {roi}")
        mom_voxel_list = []
        hemisphere = 'l' if roi % 2 == 0 else 'r'
        
        # Determine the number of trials by counting the files
        # trial_path = os.path.join(BASE_PATH, subject)
        # trial_num = len([name for name in os.listdir(trial_path) if os.path.isfile(os.path.join(trial_path, name)) and name.endswith('.mat')])
        
        for num in tqdm(range(1, self.cfg.TRIAL_NUM + 1), desc=f"Loading trials for {subject}"):
            try:
                file_path = f'{self.paths}{self.stage}/{self.paradigm}/{subject}/{str(roi)}/{subject}_{self.paradigm}_{self.stage}_voxel_{num}_{hemisphere}.mat'
                mom_voxel = loadmat(file_path)['momint_1']
                
                # Apply bandpass filter
                data_filter = eeg_bp_filter(mom_voxel[:, :self.cfg.time_points], fs=self.cfg.sample_rate, freqb=self.freq_band)
                mom_voxel_list.append(data_filter)
            except Exception as e:
                logging.error(f"Error processing trial {num}: {str(e)}")
                continue
        
        return mom_voxel_list

    @staticmethod
    def apply_threshold_and_smooth(mom_voxel_list, threshold=1):
        """Apply threshold and smoothing to the data."""
        mom_temp = np.concatenate(mom_voxel_list, 1)
        
        # Find appropriate threshold
        for thres in range(int(np.mean(np.abs(mom_temp), 1).min()), int(np.mean(np.abs(mom_temp), 1).max())):
            voxels_idx = np.mean(np.abs(mom_temp), 1) >= thres
            if np.sum(voxels_idx) / mom_temp.shape[0] <= threshold:
                return [smooth_average(data[voxels_idx, :], 3, 3) for data in mom_voxel_list]
        
        return mom_voxel_list

class PCAVarianceAnalyzer(EEGAnalyzer):
    def analyze(self):
        """Mode 1: PCA analysis of variance"""
        logging.info("Starting PCA variance analysis...")
        variance_results = []
        variance_results_sum = []
        for roi in tqdm(self.cfg.ROIS, desc="Processing ROIs"):
            roi_variance_sum = []
            roi_variance = []
            for subject in self.cfg.SUBJECTS:
                try:
                    # Load and process data
                    mom_voxel_list = self.load_and_process_data(subject, roi)
                    logging.info(f"Applying threshold and smoothing for {subject}")
                    
                    # Apply threshold and smoothing
                    mom_avg_list = self.apply_threshold_and_smooth(mom_voxel_list)
                    
                    # Apply Gaussian smoothing
                    win = norm_gauss_window(0.03, 0.05)
                    mom_smooth_list = [smooth_data(data.T, win=win, backend='convolve1d')[10:40, :].T 
                                    for data in mom_avg_list]
                    
                    # Calculate PCA variance
                    data_pca, var_ratio_sum, var_ratio = get_data_mat(mom_smooth_list, 20)
                    pca_path = f"{self.save_path}pca_data/{self.stage}/{self.paradigm}/{subject}/{roi}/"
                    os.makedirs(pca_path, exist_ok=True)
                    np.save(f"{pca_path}{subject}_pca_trial_{self.freq_band}.npy", data_pca)
                    roi_variance_sum.append(var_ratio_sum)
                    roi_variance.append(var_ratio)
                    logging.info(f"Completed processing for subject {subject}, ROI {roi}")
                except Exception as e:
                    logging.error(f"Error processing subject {subject}: {str(e)}")
                    continue
            variance_results_sum.append(roi_variance_sum)
            variance_results.append(roi_variance)
            
        self.visualize_variance(variance_results_sum)
        return variance_results
    
    def visualize_variance(self, variance_data):
        """
        Visualize PCA variance results.
        
        Args:
            variance_data (list): List of variance data for each ROI
            save_path (str): Path to save the figure
        """
        fig, ax = plt.subplots(ncols=1)
        
        for i, var in enumerate(variance_data):
            var_temp = np.reshape(np.array(var), (-1, np.array(var).shape[-1]))
            shaded_errorbar(ax, np.arange(1, 21), var_temp.T, label=self.cfg.ROI_LABELS[i])
        
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlabel('Principal Components', fontdict={'size': 15})
        ax.set_ylabel('Sum of Explained Variances', fontdict={'size': 15})
        ax.set_title(f"{self.paradigm}-{self.freq_band}", fontdict={'size': 15})
        ax.set_xticks(np.arange(2, 21, 2))
        fig.tight_layout()
        
        fig.savefig(f"{self.figure_path}{self.paradigm}_{self.stage}_{self.freq_band}_var.png", format='png')
        # fig.savefig(f"{self.cfg.PATHS['figures']}pca_variance.eps", format='eps')
        # plt.show()

        
class FMACorrelationAnalyzer(EEGAnalyzer):
    def analyze_fma_correlation(self):
        """Mode 2: FMA correlation analysis"""
        # load FMA score
        df = pd.read_excel('./subj_info.xlsx')
        fma_diff = df[df['name'].isin(self.cfg.SUBJECTS['FMA'])]['FMA_Post'] - df['FMA_Pre']
        
        # Calculate the variance change for each ROI
        var_changes = []
        for roi in self.cfg.ROIS:
            roi_var = []
            for subject in self.cfg.SUBJECTS:
                pre_pca_path = f"{self.save_path}pca_data/pre/{self.paradigm}/"
                pre_data = np.load(f"{pre_pca_path}{subject}_pca_trial_{self.freq_band}.npy")
                post_pca_path = f"{self.save_path}pca_data/post/{self.paradigm}/"
                post_data = np.load()
                var_diff = self.calculate_pca(post_data, 20) - self.calculate_pca(pre_data, 20)
                roi_var.append(var_diff[:self.cfg.pc_components])
            var_changes.append(roi_var)
        
        
        correlations = []
        for roi_var in var_changes:
            corr = spearmanr(np.mean(roi_var, axis=1), fma_diff)
            correlations.append(corr.correlation)
        self._plot_correlations(correlations)

    def _plot_correlations(self, corrs):
        
        fig, ax = plt.subplots()
        ax.bar(self.cfg.ROI_LABELS, corrs)
        ax.set_xticklabels(self.cfg.ROI_LABELS, rotation=45, ha='right')
        ax.set_title('FMA Correlation')
        plt.tight_layout()
        plt.show()

class DifferenceAnalyzer(EEGAnalyzer):
    def analyze_stage_difference(self, pre_variance, post_variance):
        """Mode 3: Stage difference analysis"""
        var_pre = np.array(pre_variance)
        print(f"Pre variance shape: {var_pre.shape}")
        var_post = np.array(post_variance)
        print(f"Post variance shape: {var_post.shape}")
        var_diff = var_pre - var_post
        p_values = []
        for roi_idx in range(len(self.cfg.ROIS)):
    
            pre_flat = np.reshape(var_pre[roi_idx, :, :],-1)
            post_flat = np.reshape(var_post[roi_idx, :, :],-1)
            _, p = stats.wilcoxon(pre_flat, post_flat)
            p_values.append(p)

        labels = [f"{label}{'*' if p < 0.05 else ''}" for label, p in zip(self.cfg.ROI_LABELS, p_values)]
        self._plot_stage_diff(var_diff, labels)
        self.create_stacked_bar(var_pre, var_post, self.stage)

    def _plot_stage_diff(self, diff_data, labels):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # plot 10 components
        for roi_idx in range(len(self.cfg.ROIS)):
            ax.plot(
                np.mean(diff_data[roi_idx], axis=0)[:10],  
                label=labels[roi_idx],
                marker='o',
                markersize=5,
                linewidth=2
            )
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Principal Components', fontsize=14)
        ax.set_ylabel('Variance Difference (Pre - Post)', fontsize=14)
        ax.set_title(f'{self.paradigm}-{self.freq_band}', fontsize=16)
        ax.set_xticks(np.arange(0, 10))
        ax.set_xticklabels(np.arange(1, 11))
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.grid(alpha=0.3, linestyle='--')
        
        # save figure
        fig.tight_layout()
        save_path = f"{self.figure_path}{self.paradigm}_{self.freq_band}_stage_diff.png"
        fig.savefig(save_path, format='png', dpi=1000)
        plt.close(fig)

    def create_stacked_bar(self, pre, post, stage_name):
        """Create a stacked bar plot of the variance data."""
        pc_num = self.cfg.PC_NUM  
        components = [f'PC{i+1}' for i in range(pc_num)] + ['Others']
        print(pre.shape)
        print(post.shape)
        pre_var = np.concatenate((pre[:, :,:pc_num],np.sum(pre[:,:,pc_num:],axis=-1,keepdims=True)),axis=-1)
        var_pre_avg = np.mean(pre_var,axis=1)
        print(var_pre_avg.shape)
        post_var = np.concatenate((post[:, :,:pc_num],np.sum(post[:,:,pc_num:],axis=-1,keepdims=True)),axis=-1)
        var_post_avg = np.mean(post_var,axis=1)
        print(var_post_avg.shape)
        # 平均处理
        # avg_data = np.mean(merged_data, axis=1)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12,6), dpi=300)
        bottom = np.zeros(len(self.cfg.ROI_LABELS))
        bottom_vals_pre = np.zeros(len(self.cfg.ROI_LABELS))
        bottom_vals_post = np.zeros(len(self.cfg.ROI_LABELS))
        
        colors = np.array([(75,116,178),(144,190,224),(230,241,243),(255,223,146),(252,140,90),(219,49,36)])/255
        x = np.arange(0, len(self.cfg.ROIS))+1
        width = 0.45
        for i in range(var_pre_avg.shape[-1]):
            rects1 = ax.bar(x - width/2 - 0.01, var_pre_avg[:,i], width=width,bottom=bottom_vals_pre,
                            label=components[i], color=colors[i], edgecolor='none')
            bottom_vals_pre += var_pre_avg[:,i]
            rects2 = ax.bar(x + width/2 + 0.01, var_post_avg[:, i], width=width, bottom=bottom_vals_post,
                            color=colors[i], edgecolor='none')
            bottom_vals_post += var_post_avg[:, i]
        # ax.set_ylim([0,1.01])
        # plt.show()
        ax.set_xticks(x)
        ax.set_xticklabels(self.cfg.ROI_LABELS,fontsize=12)
        ax.set_xlabel('Regions of Interest', fontsize=15)
        ax.set_ylabel('Explained Variance(%)', fontsize=15)
        plt.grid(axis='y',alpha=0.5,ls='--')
        plt.legend(frameon=False, bbox_to_anchor=(1.01,1), fontsize=12)
        plt.tight_layout()
        fig.savefig(
            f"{self.figure_path}{stage_name}_stacked_bar.png",
            format='png', 
            bbox_inches='tight',
            dpi=1000
        )
        plt.show()
        # plt.show()
        # plt.close(fig)


    # def _plot_variance(self, data):
        
    #     fig, ax = plt.subplots(figsize=(10,6))
    #     for i, roi_data in enumerate(data):
    #         avg = np.mean(roi_data, axis=0)
    #         shaded_errorbar(ax, np.arange(1, len(avg)+1), avg, 
    #                        label=self.cfg.ROI_LABELS[i])
    #     ax.set(xlabel='Principal Components', ylabel='Explained Variance',
    #           title=f'{self.cfg.PARADIGM}-{self.cfg.FREQ_BAND}')
    #     ax.legend()
    #     fig.savefig(f"{self.cfg.PATHS['figures']}pca_variance.png", format='png')
    #     # fig.savefig(f"{self.cfg.PATHS['figures']}pca_variance.eps", format='eps')
    #     plt.show()


# ================== MAIN ==================
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

    
    config = Config(mode='PCA_VAR')  # PCA_VAR, FMA_CORR, STAGE_DIFF
    for stage in config.STAGES:
        for paradigm in config.PARADIGMS:
            for freq_band in config.FREQ_BANDS:
                analyzer = PCAVarianceAnalyzer(config, stage, paradigm, freq_band)
                variance = analyzer.analyze()
    # pre_analyzer = PCAVarianceAnalyzer(config, 'pre', 'AO1', 'alpha')
    # pre_variance = pre_analyzer.analyze()

    # post_analyzer = PCAVarianceAnalyzer(config, 'post', 'AO1', 'alpha')
    # post_variance = post_analyzer.analyze()

    # diff_analyzer = DifferenceAnalyzer(config, 'pre', 'AO1', 'alpha')
    # diff_analyzer.analyze_stage_difference(pre_variance, post_variance)
    # if config.mode == 'PCA_VAR':
    #     analyzer.analyze_pca_variance()
    # elif config.mode == 'FMA_CORR':
    #     analyzer.analyze_fma_correlation()
    # elif config.mode == 'STAGE_DIFF':
    #     analyzer.analyze_stage_difference()
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from utils import canoncorr, divide_pair
from config.config import Config, DataLoader


class Analysis:
    """Performs CCA and statistical analysis"""
    @staticmethod
    def compute_cca_scores(data_list: List[np.ndarray]) -> List[float]:
        """Compute CCA scores between subject pairs"""
        scores = []
        subj_pairs = divide_pair(data_list)
        for pair in subj_pairs:
            r = canoncorr(data_list[pair[0]], data_list[pair[1]], fullReturn=False)
            scores.append(r)
        return scores
    
    @staticmethod
    def compute_correlation_coefficients(data_list: List[np.ndarray]) -> np.ndarray:
        """Compute correlation coefficients between subject pairs"""
        coef_pairs = []
        subj_pairs = divide_pair(data_list)
        
        for pair in subj_pairs:
            coeffs = []
            for dim in range(data_list[pair[0]].shape[-1]):
                r = stats.pearsonr(data_list[pair[0]][:, dim],
                                 data_list[pair[1]][:, dim]).statistic
                coeffs.append(r)
            coef_pairs.append(np.array(coeffs))
        return np.array(coef_pairs)


class Visualization:
    """Handles plotting and visualization"""
    def __init__(self, config: Config):
        self.config = config
        
    def plot_cca_results(self, scores: List[np.ndarray], freq: str):
        """Plot CCA results for all ROIs"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, score in enumerate(scores):
            self._add_shaded_errorbar(ax, score, self.config.ROI_LABELS[i])
            
        self._setup_plot(ax, 'Neural Modes', 'Canonical Correlation',
                        f"{self.config.PARADIGM}-{freq}")
        plt.show()
        self._save_figure(fig, f'cross_subj_CCA_all_{self.config.PARADIGM}_{freq}')

    def _add_shaded_errorbar(self, ax, data: np.ndarray, label: str):
        """Add shaded error bar to plot"""
        x = np.arange(1, data.shape[1] + 1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean-std, mean+std, alpha=0.2)

    def _setup_plot(self, ax, xlabel: str, ylabel: str, title: str):
        """Setup plot formatting"""
        ax.legend(fontsize=10)
        ax.set_xlabel(xlabel, fontdict={'size': 15})
        ax.set_ylabel(ylabel, fontdict={'size': 15})
        ax.set_xticks(np.arange(2, 21, 2))
        ax.set_title(title, fontdict={'size': 15})
        ax.set_ylim([0, 0.9])

    def _save_figure(self, fig, name: str):
        """Save figure to file"""
        save_path = '/analysis/results/difference_analysis' / f'{name}.eps'
        fig.savefig(save_path, format='eps', dpi=1000)



def main():
    """Main execution function"""
    config = Config(True)
    data_loader = DataLoader(config)
    analysis = Analysis()
    viz = Visualization(config)
    # Process each frequency band
    for freq in config.FREQUENCY_BANDS:
        cca_scores = []
        correlation_coeffs = []
        
        # Process each ROI
        for roi in config.ROIS:
            # Load data for all subjects
            subj_data = []
            for subj in config.PATH.iterdir():
                if subj.is_dir():
                    data = data_loader.load_subject_data(subj.name, roi, freq)
                    subj_data.append(data)
            
            # Compute analysis scores
            cca_score = analysis.compute_cca_scores(subj_data)
            corr_coeffs = analysis.compute_correlation_coefficients(subj_data)
            
            cca_scores.append(cca_score)
            correlation_coeffs.append(corr_coeffs)
        
        # Visualize results
        viz.plot_cca_results(cca_scores, freq)

if __name__ == "__main__":
    main()
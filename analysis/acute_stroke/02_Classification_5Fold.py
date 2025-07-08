import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC,SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils import *
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
import pickle
import pandas as pd

class Config:
    stroke_data_path = 'stroke_data'
    rest_data_path = 'rest_data'
    rng = np.random.default_rng(np.random.SeedSequence(42))
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
    ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
    # ROIs = [1]
    train_stage = ['high','low']
    Paradigm = ['AO','rest']
    pcNum = 4
    rest_trial = 13
    classifier_model = GaussianNB
    classifier_params = {}

def load_data(cf):
    high_subj_path = os.path.join(cf.stroke_data_path, 'high')
    low_subj_path = os.path.join(cf.stroke_data_path, 'low')
    high_subj = os.listdir(high_subj_path)
    low_subj = os.listdir(low_subj_path)
    subj_list = []
    subj_list.append(high_subj)
    subj_list.append(low_subj)
    return subj_list



def classification_task(cf, subj_list, freq, classifier_model):
    stage_scores = []
    for i in range(len(cf.train_stage)):
        CCA_score_stage = []
        roi_scores = []

        for roi in cf.ROIs:
            data_list = []
            label_list = []
            for subj in subj_list[i]:
                data_list_ = []
                label_list_ = []

                AO_Path = os.path.join(cf.stroke_data_path, cf.train_stage[i], subj, f'roi_{roi}', f'{subj}_{cf.Paradigm[0]}_pca_trial_{freq}.npy')
                data_AO = np.load(AO_Path)
                data_list_.append(data_AO)
                label_list_.append(np.ones((data_AO.shape[0], 1)))

                rest_path = os.path.join(cf.rest_data_path, cf.train_stage[i], subj, f'roi_{roi}', f'{subj}_{cf.Paradigm[1]}_pca_trial_{freq}.npy')
                data_rest = np.load(rest_path)
                data_list_.append(data_rest[:cf.rest_trial, :, :])
                label_list_.append(np.zeros((cf.rest_trial, 1)))
                data_list_ = np.vstack(data_list_)
                label_list_ = np.vstack(label_list_)
                rank = min(np.linalg.matrix_rank(data_list_))
                data_list.append(data_list_[:, :, :rank])
                label_list.append(label_list_)
            
            rank_min = min([data_tmp.shape[-1] for data_tmp in data_list])
            subj_pair = divide_pair(data_list)
            n_time = data_list_.shape[1]
            n_comp = rank_min

            allData = [data[:,:,:n_comp] for data in data_list]
            allData = np.concatenate(allData,axis=0).reshape((-1, n_time * n_comp))
            label = np.squeeze(np.concatenate(label_list,axis=0))

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            scores = []
            for train_index, test_index in skf.split(allData, label):
                X_train, X_test = allData[train_index], allData[test_index]
                y_train, y_test = label[train_index], label[test_index]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                classifier = classifier_model(**cf.classifier_params)

                classifier.fit(X_train_scaled, y_train)

                scores.append(classifier.score(X_test_scaled, y_test))
            roi_scores.append(scores)
        stage_scores.append(roi_scores)
    return stage_scores

def statictical_analysis_and_visualize_result(cf, stage_scores, freq, classifier_model, save_path):
    
    effectSize_cls5fold = []
    power_cls5fold = []
    sign_diff = np.squeeze(np.zeros((1,len(cf.ROIs))))
    p_value_5fold = []
    for roi_num in range(len(cf.ROIs)):
        s, p = stats.wilcoxon(stage_scores[0][roi_num], stage_scores[1][roi_num])
        p_value_5fold.append(p)
        if p < 0.05:
            sign_diff[roi_num] = 1

        effectSize_cls5fold.append(cohens_d(stage_scores[0][roi_num], stage_scores[1][roi_num]))
        pow_, _ = stat_power(effectSize_cls5fold[roi_num], sample_size=len(stage_scores[0][roi_num]))
        power_cls5fold.append(pow_)
    
    write_path = os.path.join(save_path, 'classification')
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    # Create a DataFrame with all the metrics
    results_df = pd.DataFrame({
        'p_values': p_value_5fold,
        'cohens_d': effectSize_cls5fold,
        'power': power_cls5fold
    })
    
    # Save to CSV file
    results_df.to_csv(write_path+f'/AORestCls5FoldDiff_{freq}_{str(classifier_model)}.csv', index=False)

    # Visualization
    y_cls = []
    y_cls_std = []
    for i in range(len(stage_scores)):
        y_cls_temp = []
        std_temp = []
        for ii in range(len(cf.ROIs)):
            y_cls_temp.append(np.mean(stage_scores[i][ii]))
            std_temp.append(np.std(stage_scores[i][ii]))
        y_cls.append(y_cls_temp)
        y_cls_std.append(std_temp)
    y_cls = np.array(y_cls)
    y_cls_std = np.array(y_cls_std)

    fig,ax = plt.subplots(ncols=1)
    x = np.arange(0, len(cf.ROIs))+1
    width = 0.4
    rects1 = ax.bar(x - width/2, y_cls[0,:], width, label='High NIHSS',color=['#B39CD8'])
    ax.errorbar(x - width/2, y_cls[0,:], yerr=y_cls_std[0,:], ecolor='red', fmt='.',
                markerfacecolor='#B39CD8', markeredgecolor='#B39CD8', elinewidth=1.5,capsize=5)
    rects2 = ax.bar(x + width/2, y_cls[1,:], width, label='Low NIHSS',color=['#F5CBA7'])
    ax.errorbar(x + width/2, y_cls[1,:], yerr=y_cls_std[1,:], ecolor='red', fmt='.',
                markerfacecolor='#F5CBA7', markeredgecolor='#F5CBA7', elinewidth=1.5, capsize=5)
    sign = np.squeeze(np.ones((1,len(cf.ROIs))))
    ax.scatter(np.arange(0,len(cf.ROIs))[sign_diff==1], sign[sign_diff==1], marker='*', c='r')
    ax.set_xticks(np.arange(0, len(cf.ROIs), 1)+1)
    ax.set_xticklabels(cf.ROIs_label, rotation=15)
    ax.set_ylim([0.4,0.9])
    ax.set_xlabel('Regions of Interest', fontdict={'size':15})
    ax.set_ylabel('Accuracy', fontdict={'size':15})
    ax.tick_params(labelsize=12)
    ax.set_title(freq+'-Classification Performance', fontdict={'size':15})
    ax.legend(fontsize=15)
    fig.tight_layout()
    figsave_path = os.path.join(save_path, 'figure', 'Classification')
    if not os.path.exists(figsave_path):
        os.makedirs(figsave_path)
    fig.savefig(figsave_path+f'/{cf.Paradigm[0]}_{cf.Paradigm[1]}_{freq}_{str(classifier_model)}.png', format='png', dpi=1000)
    fig.savefig(figsave_path+f'/{cf.Paradigm[0]}_{cf.Paradigm[1]}_{freq}_{str(classifier_model)}.eps', format='eps', dpi=1000)
def main():
    cf = Config()
    freqb = ['beta', 'alpha','theta','delta']
    classifier_model = [RandomForestClassifier]
    save_path = 'analysis_result/powerAndEffectSize'
    subj_list = load_data(cf)
    for f in freqb:
        for model in classifier_model:
            stage_scores = classification_task(cf, subj_list, f, model)
            statictical_analysis_and_visualize_result(cf, stage_scores, f, model, save_path)

if __name__ == "__main__":
    main()

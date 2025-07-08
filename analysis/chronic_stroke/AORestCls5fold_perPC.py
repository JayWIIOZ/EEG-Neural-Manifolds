import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC,SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from Code.utils import *
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(np.random.SeedSequence(42))
ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
# ROIs = [1]
train_stage = ['pre','post']
Paradigm = ['AO1','rest']
freqb = 'beta'
pcNum = 4
rest_trial = 13

# classifier_model = LinearSVC
# classifier_params = {'max_iter':10000}
classifier_model = RandomForestClassifier
classifier_params = {}

stage_scores = []
for trainStage in train_stage:
    # load data
    CCA_score_stage = []
    AO_load_path = 'G:/CUHK_intern\RESULTS\Multimodality/' + trainStage + '/' + Paradigm[0] + '/'
    rest_load_path = 'G:/CUHK_intern\RESULTS\Multimodality/' + trainStage + '/' + Paradigm[1] + '/'
    # subj_list = os.listdir(load_path)
    subj_list = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wsc', 'wwf']
    roi_scores = []
    for roi in ROIs:
        data_list = []
        label_list = []
        for subj in subj_list:
            data_list_ = []
            label_list_ = []

            AO_path = AO_load_path + subj + '/trial/' + str(roi) + '/'
            data_AO = np.load(AO_path + subj + '_' + Paradigm[0] + '_' + trainStage + '_pca_trial_' + freqb + '.npy')
            data_list_.append(data_AO)
            label_list_.append(np.ones((data_AO.shape[0],1)))

            rest_path = rest_load_path + subj + '/trial/' + str(roi) + '/'
            data_rest = np.load(rest_path + subj + '_' + Paradigm[1] + '_' + trainStage + '_pca_trial_' + freqb + '.npy')
            data_list_.append(data_rest[:rest_trial,:,:])
            label_list_.append(np.zeros((rest_trial,1)))
            data_list_ = np.vstack(data_list_)
            label_list_ = np.vstack(label_list_)
            rank = min(np.linalg.matrix_rank(data_list_))
            data_list.append(data_list_[:,:,:rank])
            label_list.append(label_list_)

        rank_min = min([data_tmp.shape[-1] for data_tmp in data_list])
        subj_pair = divide_pair(data_list)
        n_time = data_list_.shape[1]
        n_comp = rank_min

        comp_scores = []
        for pc_num in range(n_comp):
            allData = [data[:,:,pc_num] for data in data_list]
            allData = np.concatenate(allData,axis=0).reshape((-1, n_time * 1))
            label = np.squeeze(np.concatenate(label_list,axis=0))

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            scores = []
            for train_index, test_index in skf.split(allData, label):
                X_train, X_test = allData[train_index], allData[test_index]
                y_train, y_test = label[train_index], label[test_index]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                classifier = classifier_model(**classifier_params)
                classifier.fit(X_train_scaled, y_train)

                scores.append(classifier.score(X_test_scaled, y_test))
            comp_scores.append(scores)
        roi_scores.append(comp_scores)
    stage_scores.append(roi_scores)

# visualization
y_diff = []
y_diff_std = []
for i in range(len(ROIs)):
    y_diff_temp = []
    std_temp = []
    min_comp = min(len(stage_scores[0][i]),len(stage_scores[1][i]))
    for ii in range(min_comp):
        y_diff_temp.append(np.mean(np.array(stage_scores[0][i][ii])-np.array(stage_scores[1][i][ii])))
        std_temp.append(np.std(np.array(stage_scores[0][i][ii])-np.array(stage_scores[1][i][ii])))
    y_diff.append(y_diff_temp)
    y_diff_std.append(std_temp)


# ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
# fig,ax = plt.subplots(ncols=1)
# x = np.arange(0, len(ROIs))+1
# width = 0.4
# rects1 = ax.bar(x - width/2, y_cls[0,:], width, label='Pre',color=['#B39CD8'])
# ax.errorbar(x - width/2, y_cls[0,:], yerr=y_cls_std[0,:], ecolor='red', fmt='.',
#             markerfacecolor='#B39CD8', markeredgecolor='#B39CD8', elinewidth=1.5,capsize=5)
# rects2 = ax.bar(x + width/2, y_cls[1,:], width, label='Post',color=['#F5CBA7'])
# ax.errorbar(x + width/2, y_cls[1,:], yerr=y_cls_std[1,:], ecolor='red', fmt='.',
#             markerfacecolor='#F5CBA7', markeredgecolor='#F5CBA7', elinewidth=1.5, capsize=5)
# sign = np.squeeze(np.ones((1,len(ROIs))))
# ax.scatter(np.arange(0,len(ROIs))[sign_diff==1], sign[sign_diff==1], marker='*', c='r')
# ax.set_xticks(np.arange(0, len(ROIs), 1)+1)
# ax.set_xticklabels(ROIs_label, rotation=15)
# ax.set_ylim([0.4,1])
# ax.set_xlabel('Regions of Interest', fontdict={'size':15})
# ax.set_ylabel('Accuracy', fontdict={'size':15})
# ax.tick_params(labelsize=12)
# ax.set_title(freqb+'-Classification Performance', fontdict={'size':15})
# ax.legend(fontsize=15)
# fig.tight_layout()
# plt.show()
# fig.savefig('G:\CUHK_Intern\RESULTS/figure\Multimodality/PrePost/' + 'AO_Rest_cls_blanceTrial_allComp_' + freqb + '.eps', format='eps', dpi=1000)
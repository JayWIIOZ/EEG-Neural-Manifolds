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
import pickle

rng = np.random.default_rng(np.random.SeedSequence(12345))
ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
# ROIs = [1]
train_stage = ['pre','post']
Paradigm = ['AO1','rest']
freqb = 'beta'
pcNum = 4
rest_trial = 13

classifier_model = LinearSVC
classifier_params = {'max_iter':10000}
# classifier_model = RandomForestClassifier
# classifier_params = {}

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
        for subj in subj_list:
            data_list_ = []

            AO_path = AO_load_path + subj + '/trial/' + str(roi) + '/'
            data_AO = np.load(AO_path + subj + '_' + Paradigm[0] + '_' + trainStage + '_pca_trial_' + freqb + '.npy')
            data_list_.append(data_AO)

            rest_path = rest_load_path + subj + '/trial/' + str(roi) + '/'
            data_rest = np.load(rest_path + subj + '_' + Paradigm[1] + '_' + trainStage + '_pca_trial_' + freqb + '.npy')
            data_list_.append(data_rest[:rest_trial,:,:])
            data_list_ = np.vstack(data_list_)
            rank = min(np.linalg.matrix_rank(data_list_))
            data_list.append(data_list_[:,:,:rank])

        rank_min = min([data_tmp.shape[-1] for data_tmp in data_list])
        subj_pair = divide_pair(data_list)
        n_time = data_list_.shape[1]
        n_comp = rank_min

        pair_scores = []
        for pair_num in range(len(subj_pair)):
            temp_ind = subj_pair[pair_num]
            trial_min = min(data_list[temp_ind[0]].shape[0], data_list[temp_ind[1]].shape[0])

            temp1 = data_list[temp_ind[0]][-trial_min:,:,:rank_min].reshape((-1,n_comp))
            temp2 = data_list[temp_ind[1]][-trial_min:,:,:rank_min].reshape((-1,n_comp))

            A, B, *_ = canoncorr(temp1, temp2, fullReturn=True)
            X1_test = temp1 @ A @ np.linalg.inv(B)
            X2_test = temp2 @ B @ np.linalg.inv(A)
            X1_test = X1_test.reshape((-1, n_time * n_comp))
            X2_test = X2_test.reshape((-1, n_time * n_comp))

            scores = []
            for subj_num in range(len(subj_pair[pair_num])):
                X = data_list[temp_ind[subj_num]][-trial_min:,:,:rank_min].reshape((trial_min,-1))
                Y = np.squeeze(np.hstack([np.ones((1, X.shape[0]-rest_trial)), 2*np.ones((1, rest_trial))]))

                # shuffle
                trial_index1 = np.arange(Y.shape[-1])
                # to guarantee shuffled ids
                while ((all_id_sh := rng.permutation(trial_index1)) == trial_index1).all():
                    continue
                trial_index1 = all_id_sh
                X_train, Y_train = X[trial_index1, :], Y[trial_index1]

                # min_max_scaler = MinMaxScaler()
                # X_train_scale = min_max_scaler.fit_transform(X_train)

                classifier = classifier_model(**classifier_params)
                classifier.fit(X_train, Y_train)

                rng.shuffle(trial_index1)
                if subj_num == 0:
                    X_test = X2_test[trial_index1,:]
                else:
                    X_test = X1_test[trial_index1, :]
                Y_test = Y[trial_index1]

                # X_test_scale = min_max_scaler.fit_transform(X_test)

                scores.append(classifier.score(X_test, Y_test))
            pair_scores.append(np.mean(scores))
        roi_scores.append(pair_scores)
    stage_scores.append(roi_scores)

# statistical test
effectSize_cls = []
power_cls = []
sign_diff = np.squeeze(np.zeros((1,len(ROIs))))
p_value = []
for roi_num in range(len(ROIs)):
    s, p = stats.wilcoxon(stage_scores[0][roi_num], stage_scores[1][roi_num])
    p_value.append(p)
    if p < 0.05:
        sign_diff[roi_num] = 1

    effectSize_cls.append(cohens_d(stage_scores[0][roi_num], stage_scores[1][roi_num]))
    pow_, _ = stat_power(effectSize_cls[roi_num], sample_size=len(stage_scores[0][roi_num]))
    power_cls.append(pow_)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\classification/AORestClsDiff_pvalues_'+freqb+'.txt', 'w') as f:
    for item in p_value:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\classification/AORestClsDiff_cohensd_'+freqb+'.txt', 'w') as f:
    for item in effectSize_cls:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\classification/AORestClsDiff_power_'+freqb+'.txt', 'w') as f:
    for item in power_cls:
        f.write(f"{item}\n")

# visualization
y_cls = []
y_cls_std = []
for i in range(len(stage_scores)):
    y_cls_temp = []
    std_temp = []
    for ii in range(len(ROIs)):
        y_cls_temp.append(np.mean(stage_scores[i][ii]))
        std_temp.append(np.std(stage_scores[i][ii]))
    y_cls.append(y_cls_temp)
    y_cls_std.append(std_temp)
y_cls = np.array(y_cls)
y_cls_std = np.array(y_cls_std)

ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
fig,ax = plt.subplots(ncols=1)
x = np.arange(0, len(ROIs))+1
width = 0.4
rects1 = ax.bar(x - width/2, y_cls[0,:], width, label='Pre',color=['#82B0D2'])
ax.errorbar(x - width/2, y_cls[0,:], yerr=y_cls_std[0,:], ecolor='red', fmt='.',
            markerfacecolor='#82B0D2', markeredgecolor='#82B0D2', elinewidth=1.5,capsize=5)
rects2 = ax.bar(x + width/2, y_cls[1,:], width, label='Post',color=['#FA7F6F'])
ax.errorbar(x + width/2, y_cls[1,:], yerr=y_cls_std[1,:], ecolor='red', fmt='.',
            markerfacecolor='#FA7F6F', markeredgecolor='#FA7F6F', elinewidth=1.5, capsize=5)
sign = np.squeeze(np.ones((1,len(ROIs))))
ax.scatter(np.arange(0,len(ROIs))[sign_diff==1], sign[sign_diff==1], marker='*', c='r')
ax.set_xticks(np.arange(0, len(ROIs), 1)+1)
ax.set_xticklabels(ROIs_label, rotation=15)
ax.set_ylim([0.4,1])
ax.set_xlabel('Regions of Interest', fontdict={'size':15})
ax.set_ylabel('Accuracy', fontdict={'size':15})
ax.tick_params(labelsize=12)
ax.set_title(freqb+'-Classification Performance', fontdict={'size':15})
ax.legend(fontsize=15)
fig.tight_layout()
plt.show()
# fig.savefig('G:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'AO_Rest_cls_blanceTrial_' + freqb + '.eps', format='eps', dpi=1000)


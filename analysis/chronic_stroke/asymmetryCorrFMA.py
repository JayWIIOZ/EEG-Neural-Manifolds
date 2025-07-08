import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from utils import *
from scipy.spatial import distance
import pickle

'''
Asymmetry better to regress with BSI as well as FMA scores
'''

hemisphere = [[1,2],[19,20],[59,60],[61,62]]
train_stage = ['pre','post']
Paradigm = 'rest'
freqb = 'theta'

# load FMA scores
df = pd.read_excel('G:/CUHK_Intern/subj_info.xlsx')
subj_fma = [[df.name],[df['FMA_Pre']],[df['FMA_Post']]]
# subj_list = df['name'].tolist()
subj_list = ['kmt','wws','nsk','nwc','ock','wsc','wwf']

# subj_list = ['kmt','wws','nsk','nwc','ock','wsc','wwf']
CCA_score = []
for roi_num in range(len(hemisphere)):
    CCA_score_roi = []
    data_stage_list_r = []
    data_stage_list_l = []
    for trainStage in train_stage:
        load_path = 'G:/CUHK_Intern/RESULTS/Multimodality/'+trainStage+'/'+Paradigm+'/'
        data_tphate_list_r = []
        data_tphate_list_l = []
        for subj in subj_list:
            # mom_decom_r = loadmat(load_path+subj+'/'+subj+'_'+Paradigm+'_'+trainStage+'_decompose_'+
            #                           freqb+'_'+str(hemisphere[roi_num][0])+'_r.mat')['mom_decom']

            # mom_decom_l = loadmat(load_path+subj+'/'+subj+'_'+Paradigm+'_'+trainStage+'_decompose_'+
            #                           freqb+'_'+str(hemisphere[roi_num][1])+'_l.mat')['mom_decom']

            data_path_r = load_path + subj + '/trial/' + str(hemisphere[roi_num][0]) + '/'
            data_tphate_r = np.load(
                data_path_r + subj + '_' + Paradigm + '_' + trainStage + '_pca_trial_' + freqb + '.npy')
            data_path_l = load_path + subj + '/trial/' + str(hemisphere[roi_num][1]) + '/'
            data_tphate_l = np.load(
                data_path_l + subj + '_' + Paradigm + '_' + trainStage + '_pca_trial_' + freqb + '.npy')

            time_len = data_tphate_r.shape[1]
            trial_min = min(data_tphate_r.shape[0], data_tphate_l.shape[0])
            rank = min(min(np.linalg.matrix_rank(data_tphate_r)), min(np.linalg.matrix_rank(data_tphate_l)))
            data_tphate_r = data_tphate_r[:trial_min, :, :rank]
            data_tphate_l = data_tphate_l[:trial_min, :, :rank]

            data_tphate_reshape_r = np.reshape(data_tphate_r, (-1, data_tphate_r.shape[-1]))
            data_tphate_reshape_l = np.reshape(data_tphate_l, (-1, data_tphate_l.shape[-1]))

            data_tphate_list_r.append(data_tphate_reshape_r)
            data_tphate_list_l.append(data_tphate_reshape_l)

        rank_min = min(min([data_tmp.shape[-1] for data_tmp in data_tphate_list_r]),
                       min([data_tmp.shape[-1] for data_tmp in data_tphate_list_l]))
        for i in range(len(data_tphate_list_r)):
            data_tphate_list_r[i] = data_tphate_list_r[i][:, :rank_min]
            data_tphate_list_l[i] = data_tphate_list_l[i][:, :rank_min]

        data_stage_list_r.append(data_tphate_list_r)
        data_stage_list_l.append(data_tphate_list_l)

    rank_min_ = min(data_stage_list_r[0][0].shape[-1], data_stage_list_r[1][0].shape[-1])
    for stage_num in range(2):
        CCA_score_stage = []
        for j in range(len(data_stage_list_l[stage_num])):
            r1 = canoncorr(data_stage_list_r[stage_num][j][:, :rank_min_],
                           data_stage_list_l[stage_num][j][:, :rank_min_], fullReturn=False)
            CCA_score_stage.append(r1)
        CCA_score_roi.append(CCA_score_stage)

    CCA_score.append(CCA_score_roi)

# obtain FMA scores
fma_scores_all = []
for i in range(2):
    fma_scores = []
    for subj in subj_list:
        fma_scores.append(subj_fma[i + 1][0][subj_fma[0][0] == subj].values)
    fma_scores_all.append(fma_scores)

# regression between difference and FMA scores
fma_scores_all = np.array(fma_scores_all)
fma_diff = np.squeeze(fma_scores_all[1] - fma_scores_all[0]) # after - before

# correlation
corr_diff = []
p_diff = []
power_asymFMAdiff = []
for num in range(len(CCA_score)):
    temp = np.array(CCA_score[num])
    CCA_score_diff = temp[1] - temp[0]
    res_diff = pearsonr(np.mean(CCA_score_diff,1), fma_diff)
    pow_, _ = stat_power(res_diff.correlation, sample_size=fma_diff.shape[0])
    power_asymFMAdiff.append(pow_)
    corr_diff.append(res_diff.correlation)
    p_diff.append(res_diff.pvalue)


with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/asymmetry/asymmetryFMA_power_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in power_asymFMAdiff:
        f.write(f"{item}\n")

with open('G:\CUHK_Intern\RESULTS/additionalResults/asymmetryCorrFMA_'+Paradigm+'_'+freqb+'_corr.txt', 'w') as f:
    for item in corr_diff:
        f.write(f"{item}\n")

with open('G:\CUHK_Intern\RESULTS/additionalResults/asymmetryCorrFMA_'+Paradigm+'_'+freqb+'_pvalue.txt', 'w') as f:
    for item in p_diff:
        f.write(f"{item}\n")

# visualization
# LABELs = ROIs_label = ['PreCG.L - PreCG.R','SMA.L - SMA.R','SPG.L - SPG.R','IPL.L - IPL.R']
# fig,ax = plt.subplots(ncols=1)
# for ii in range(len(p_diff)):
#     ax.plot(np.arange(1, len(p_diff[ii])+1),np.array(p_diff[ii]),label=LABELs[ii])
# ax.plot(np.arange(1,21),np.squeeze(0.05*np.ones((1,20))), c='r',linestyle='--',linewidth=2)
# ax.legend(fontsize=10)
# ax.set_xlabel('Canonical Components', fontdict={'size':15})
# ax.set_ylabel('P Values', fontdict={'size':15})
# ax.set_title(Paradigm+'-'+freqb+'-p values', fontdict={'size':15})
# ax.set_ylim([0, 1.1])
# fig.tight_layout()
# plt.show()
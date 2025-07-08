import numpy as np
import matplotlib.pyplot as plt
import os
from Code.utils import *
# from cca_zoo.linear import GCCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import pandas as pd
import seaborn as sns
import pickle

ROIs = [1,2,19,20,59,60,61,62]
# ROIs = [1]
train_stage = ['pre','post']
Paradigm = 'rest'
freqb = 'theta'
pcNum = 4
ccNum = 4

# load data
load_path = 'G:/CUHK_intern/RESULTS/Multimodality/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt','wws','nsk','nwc','ock','wwf','wsc']

contributions_roi = []
for roi in ROIs:
    contributions_subj = []
    data_tphate_list_pre = []
    data_tphate_list_post = []
    var_list_pre = []
    var_list_post = []
    # data_tphate_ori_list_pre = []
    # data_tphate_ori_list_post = []
    for subj in subj_list:
        data_path_pre = load_path + train_stage[0] + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/'
        data_tphate_pre = np.load(
            data_path_pre + subj + '_' + Paradigm + '_' + train_stage[0] + '_pca_trial_' + freqb + '.npy')
        var_pre = np.load(
            data_path_pre + subj + '_' + Paradigm + '_' + train_stage[0] + '_pcaVar_trial_' + freqb + '.npy')
        data_path_post = load_path + train_stage[1] + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/'
        data_tphate_post = np.load(
            data_path_post + subj + '_' + Paradigm + '_' + train_stage[1] + '_pca_trial_' + freqb + '.npy')
        var_post = np.load(
            data_path_post + subj + '_' + Paradigm + '_' + train_stage[1] + '_pcaVar_trial_' + freqb + '.npy')

        trial_min = min(data_tphate_pre.shape[0],data_tphate_post.shape[0])
        rank = min(min(np.linalg.matrix_rank(data_tphate_pre)), min(np.linalg.matrix_rank(data_tphate_post)))
        data_tphate_pre = data_tphate_pre[:trial_min, :, :rank]
        data_tphate_post = data_tphate_post[:trial_min, :, :rank]
        var_pre = var_pre[:rank]
        var_post = var_post[:rank]

        # data_tphate_ori_list_pre.append(data_tphate_pre)
        # data_tphate_ori_list_post.append(data_tphate_post)

        # compute the canonical correlation directly, to see if training cause great difference
        data_tphate_reshape_pre = np.reshape(data_tphate_pre, (-1, data_tphate_pre.shape[-1]))
        data_tphate_reshape_post = np.reshape(data_tphate_post, (-1, data_tphate_post.shape[-1]))

        # data_tphate_reshape_pre = np.mean(data_tphate_pre, axis=0)
        # data_tphate_reshape_post = np.mean(data_tphate_post, axis=0)
        data_tphate_list_pre.append(data_tphate_reshape_pre)
        data_tphate_list_post.append(data_tphate_reshape_post)
        var_list_pre.append(var_pre)
        var_list_post.append(var_post)

    rank_min = min(min([data_tmp.shape[1] for data_tmp in data_tphate_list_pre]),
                   min([data_tmp.shape[1] for data_tmp in data_tphate_list_post]))
    for i in range(len(data_tphate_list_pre)):
        data_tphate_list_pre[i] = data_tphate_list_pre[i][:,:rank_min]
        # data_tphate_list_pre[i] = data_tphate_list_pre[i][:, :]
        data_tphate_list_post[i] = data_tphate_list_post[i][:,:rank_min]
        # data_tphate_list_post[i] = data_tphate_list_post[i][:, :]
        var_list_pre[i] = var_list_pre[i][:rank_min]
        var_list_post[i] = var_list_post[i][:rank_min]

    for i in range(len(data_tphate_list_pre)):
        A, B, *_ = canoncorr(data_tphate_list_pre[i], data_tphate_list_post[i], fullReturn=True)
        weighted_contributions_pre = np.abs(A) * var_list_pre[i][:, np.newaxis]
        weighted_contributions_post = np.abs(B) * var_list_post[i][:, np.newaxis]
        contri_pre = []
        contri_post = []
        for cc_i in range(A.shape[1]):
            contri_pre.append(weighted_contributions_pre[:,cc_i] / np.sum(weighted_contributions_pre[:,cc_i]))
            contri_post.append(weighted_contributions_post[:,cc_i] / np.sum(weighted_contributions_post[:,cc_i]))

        contributions_subj.append([np.array(contri_pre),np.array(contri_post)])

    contributions_roi.append(np.array(contributions_subj)) # subject*stage*cc*pc


# visualization (proportion bar graph)
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
components = ['PC1','PC2','PC3','PC4','Others']
color = np.array([(75,101,175),(127,203,164),(233,245,161),(253,217,133),(244,111,68),(164,5,69)])/255
contributions_roi_ = []
for i in range(len(ROIs)):
    contributions_roi_.append(np.concatenate((contributions_roi[i][:,:,:ccNum,:pcNum], np.sum(contributions_roi[i][:,:,:ccNum,pcNum:],axis=-1,keepdims=True)),axis=-1))

contributions_roi_avg = np.mean(np.array(contributions_roi_),1)

x = np.arange(0, len(ROIs))+1
width = 0.45
for cc_i in range(ccNum):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 5), dpi=300)
    bottom_vals_pre = np.zeros(len(ROIs_label))
    bottom_vals_post = np.zeros(len(ROIs_label))
    var_pre_avg = contributions_roi_avg[:,0,cc_i,:]
    var_post_avg = contributions_roi_avg[:,1,cc_i,:]
    for i in range(var_pre_avg.shape[-1]):
        rects1 = ax.bar(x - width/2 - 0.01, var_pre_avg[:,i], width=width,bottom=bottom_vals_pre,
                        label=components[i], color=color[i], edgecolor='none')
        bottom_vals_pre += var_pre_avg[:,i]
        rects2 = ax.bar(x + width/2 + 0.01, var_post_avg[:, i], width=width, bottom=bottom_vals_post,
                        color=color[i], edgecolor='none', alpha=0.7)
        bottom_vals_post += var_post_avg[:, i]
    ax.set_ylim([0,1.01])
    # plt.show()
    ax.set_xticks(x)
    ax.set_xticklabels(ROIs_label,fontsize=12)
    ax.set_xlabel('Regions of Interest', fontsize=15)
    ax.set_ylabel('Principal Components Contributions', fontsize=15)
    ax.set_title('CC'+str(cc_i), fontsize=15)
    plt.grid(axis='y',alpha=0.5,ls='--')
    plt.legend(frameon=False, bbox_to_anchor=(1.01,1), fontsize=12)
    fig.tight_layout()
    save_path = 'G:\CUHK_Intern\RESULTS/additionalResults/'
    # fig.savefig(save_path + 'contributions_diff_proportion_bar_'+ Paradigm + '_' + freqb + '_' +'CC' + str(cc_i) + '.eps', dpi=1000, format='eps')
    plt.show()
    plt.close(fig)


# statistics test
p_pcr = []
effectSize_pcr = []
power_pcr = []
for cc_i in range(ccNum):
    p_cc = []
    effectSize_cc = []
    power_cc = []
    for roi_num in range(len(contributions_roi_)):
        s, p = stats.wilcoxon(contributions_roi_[roi_num][:,0,cc_i,-1], contributions_roi_[roi_num][:,1,cc_i,-1])
        p_cc.append(p)

        effectSize_cc.append(cohens_d(contributions_roi_[roi_num][:,0,cc_i,-1], contributions_roi_[roi_num][:,1,cc_i,-1]))
        pow_, _ = stat_power(effectSize_cc[roi_num], sample_size=contributions_roi_[roi_num][:,0,cc_i,-1].shape[0])
        power_cc.append(pow_)
    p_pcr.append(p_cc)
    effectSize_pcr.append(effectSize_cc)
    power_pcr.append(power_cc)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/contributionsRestPCDiff_pvalues_'+Paradigm+'_'+freqb+'.pkl', 'wb') as f:
    pickle.dump(p_pcr, f)
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/contributionsRestPCDiff_cohensd_'+Paradigm+'_'+freqb+'.pkl', 'wb') as f:
    pickle.dump(effectSize_pcr, f)
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/contributionsRestPCDiff_power_'+Paradigm+'_'+freqb+'.pkl', 'wb') as f:
    pickle.dump(power_pcr, f)




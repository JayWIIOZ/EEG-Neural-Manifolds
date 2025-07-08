import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from utils import *
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
from manifolds_distance import *
import pickle

'''
Better to conduct regression of manifolds with psd as well as erd/ers, the difference can be regressed with FMA 
alteration
'''

ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
train_stage = ['pre','post']
Paradigm = 'rest'
freqb = 'beta'

#load data
load_path = 'G:\CUHK_Intern/RESULTS/Multimodality/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt', 'wws', 'nsk', 'nwc', 'ock', 'wwf','wsc']

# load FMA scores
df = pd.read_excel('../subj_info.xlsx')
subj_fma = [[df.name],[df['FMA_Pre']],[df['FMA_Post']]]
# subj_list = df['name'].tolist()

dis_roi = []
GCCA_score_subj_roi = []
data_aligned_roi = []
for roi in ROIs:
    dis_subj = []
    data_tphate_list_pre = []
    data_tphate_list_post = []
    for subj in subj_list:
        data_path_pre = load_path + train_stage[0] + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/'
        data_tphate_pre = np.load(
            data_path_pre + subj + '_' + Paradigm + '_' + train_stage[0] + '_pca_trial_' + freqb + '.npy')
        data_path_post = load_path + train_stage[1] + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/'
        data_tphate_post = np.load(
            data_path_post + subj + '_' + Paradigm + '_' + train_stage[1] + '_pca_trial_' + freqb + '.npy')

        time_len = data_tphate_pre.shape[1]
        trial_min = min(data_tphate_pre.shape[0], data_tphate_post.shape[0])
        rank = min(min(np.linalg.matrix_rank(data_tphate_pre)), min(np.linalg.matrix_rank(data_tphate_post)))
        data_tphate_pre = data_tphate_pre[:trial_min, :, :rank]
        data_tphate_post = data_tphate_post[:trial_min, :, :rank]

        data_tphate_reshape_pre = np.reshape(data_tphate_pre, (-1, data_tphate_pre.shape[-1]))
        data_tphate_reshape_post = np.reshape(data_tphate_post, (-1, data_tphate_post.shape[-1]))

        data_tphate_list_pre.append(data_tphate_reshape_pre)
        data_tphate_list_post.append(data_tphate_reshape_post)

    rank_min = min(min([data_tmp.shape[1] for data_tmp in data_tphate_list_pre]),
                   min([data_tmp.shape[1] for data_tmp in data_tphate_list_post]))
    for i in range(len(data_tphate_list_pre)):
        data_tphate_list_pre[i] = data_tphate_list_pre[i][:, :rank_min]
        data_tphate_list_post[i] = data_tphate_list_post[i][:, :rank_min]

    data_aligned = []
    for i in range(len(data_tphate_list_pre)):
        A1, B1, r1, *_ = canoncorr(data_tphate_list_pre[i], data_tphate_list_post[i], fullReturn=True)
        U1, s1, Vh1 = svd(A1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
        U2, s2, Vh2 = svd(B1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
        temp_pre = np.reshape(data_tphate_list_pre[i] @ U1 @ Vh1,(-1, time_len, r1.shape[-1]))
        temp_post = np.reshape(data_tphate_list_post[i] @ U2 @ Vh2, (-1, time_len, r1.shape[-1]))
        data_aligned.append([np.mean(temp_pre,0),np.mean(temp_post,0)])
        # mani_diff = []
        # for trial_ii in range(temp_pre.shape[0]):
        #     temp_diff = np.array([distance.euclidean(temp_pre[trial_ii, i, :], temp_post[trial_ii, i, :]) for i in
        #                           range(temp_pre.shape[1])])
        #     mani_diff.append(np.mean(temp_diff))
        mani_diff = np.array([distance.cosine(np.mean(temp_pre,0)[ii,:], np.mean(temp_post,0)[ii,:]) for ii in range(temp_pre.shape[1])])
        dis_subj.append(np.mean(mani_diff))

    data_aligned_roi.append(data_aligned)
    dis_roi.append(dis_subj)

# obtain FMA scores
fma_scores_all = []
for i in range(2):
    fma_scores = []
    for subj in subj_list:
        fma_scores.append(subj_fma[i + 1][0][subj_fma[0][0] == subj].values)
    fma_scores_all.append(fma_scores)

fma_scores_all = np.array(fma_scores_all)
fma_diff = np.squeeze(fma_scores_all[1] - fma_scores_all[0]) # after - before

corr = []
p_value = []
power_structFMA = []
for roi_num in range(len(data_aligned_roi)):
    temp = np.array(data_aligned_roi[roi_num])
    grad = grassmann_distance(temp[:,0,:,:],temp[:,1,:,:])

    res = spearmanr(grad, fma_diff)
    corr.append(res.correlation)
    p_value.append(res.pvalue)

    pow_, _ = stat_power(res.correlation+0.0000000000001, sample_size=fma_diff.shape[0])
    power_structFMA.append(pow_)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\structure/structureDiffFMA_power_' + Paradigm + '_' + freqb + '.txt','w') as f:
    for item in power_structFMA:
        f.write(f"{item}\n")

# with open('G:\CUHK_Intern\RESULTS/additionalResults/differencePrePostCorrFMA_' + Paradigm + '_' + freqb + '_corr.txt', 'w') as f:
#     for item in corr:
#         f.write(f"{item}\n")
#
# with open('G:\CUHK_Intern\RESULTS/additionalResults/differencePrePostCorrFMA_' + Paradigm + '_' + freqb + '_pvalue.txt', 'w') as f:
#     for item in p_value:
#         f.write(f"{item}\n")

# visualization
p = np.array(p_value) < 0.05
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
fig,ax = plt.subplots(ncols=1)
ax.bar(np.arange(0, len(ROIs)),np.array(corr))
ax.scatter(np.arange(0,len(ROIs))[p==1], p[p==1],marker='*',c='r')
ax.set_xticks(np.arange(0, len(ROIs), 1))
ax.set_xticklabels(ROIs_label)
ax.set_xlabel('Regions of Interest', fontdict={'size':15})
ax.set_ylabel('Pearson Correlations', fontdict={'size':15})
ax.set_title(Paradigm+'-'+freqb+'-Correlations', fontdict={'size':15})
fig.tight_layout()
plt.show()
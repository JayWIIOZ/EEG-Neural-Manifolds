import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
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
freqb = 'beta'
pcNum = 4

# load data
load_path = 'G:/CUHK_intern/RESULTS/Multimodality/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt','wws','nsk','nwc','ock','wwf','wsc']

power_subj_roi = []
CCA_score_roi = []
GCCA_score_subj_roi = []
coef_subj_roi = []
for roi in ROIs:
    CCA_score_subj = []
    data_tphate_list_pre = []
    data_tphate_list_post = []
    # data_tphate_ori_list_pre = []
    # data_tphate_ori_list_post = []
    for subj in subj_list:
        data_path_pre = load_path + train_stage[0] + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/'
        data_tphate_pre = np.load(
            data_path_pre + subj + '_' + Paradigm + '_' + train_stage[0] + '_pca_trial_' + freqb + '.npy')
        data_path_post = load_path + train_stage[1] + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/'
        data_tphate_post = np.load(
            data_path_post + subj + '_' + Paradigm + '_' + train_stage[1] + '_pca_trial_' + freqb + '.npy')

        trial_min = min(data_tphate_pre.shape[0],data_tphate_post.shape[0])
        rank = min(min(np.linalg.matrix_rank(data_tphate_pre)), min(np.linalg.matrix_rank(data_tphate_post)))
        data_tphate_pre = data_tphate_pre[:trial_min, :, :rank]
        data_tphate_post = data_tphate_post[:trial_min, :, :rank]

        # data_tphate_ori_list_pre.append(data_tphate_pre)
        # data_tphate_ori_list_post.append(data_tphate_post)

        # compute the canonical correlation directly, to see if training cause great difference
        data_tphate_reshape_pre = np.reshape(data_tphate_pre, (-1, data_tphate_pre.shape[-1]))
        data_tphate_reshape_post = np.reshape(data_tphate_post, (-1, data_tphate_post.shape[-1]))

        # data_tphate_reshape_pre = np.mean(data_tphate_pre, axis=0)
        # data_tphate_reshape_post = np.mean(data_tphate_post, axis=0)
        data_tphate_list_pre.append(data_tphate_reshape_pre)
        data_tphate_list_post.append(data_tphate_reshape_post)

    rank_min = min(min([data_tmp.shape[1] for data_tmp in data_tphate_list_pre]),
                   min([data_tmp.shape[1] for data_tmp in data_tphate_list_post]))
    for i in range(len(data_tphate_list_pre)):
        data_tphate_list_pre[i] = data_tphate_list_pre[i][:,:rank_min]
        # data_tphate_list_pre[i] = data_tphate_list_pre[i][:, :]
        data_tphate_list_post[i] = data_tphate_list_post[i][:,:rank_min]
        # data_tphate_list_post[i] = data_tphate_list_post[i][:, :]

    for i in range(len(data_tphate_list_pre)):
        r1 = canoncorr(data_tphate_list_pre[i], data_tphate_list_post[i], fullReturn=False)
        CCA_score_subj.append(r1)

    CCA_score_roi.append(CCA_score_subj)

    # cross subject consistency before and after training
    power_stage = []
    GCCA_score_stage = []
    coef_stage = []
    # pre
    time_min = min([data_tphate.shape[0] for data_tphate in data_tphate_list_pre])
    data_tphate_list_pre_ = [data_tphate[:time_min,:] for data_tphate in data_tphate_list_pre]

    # aligned
    power = []
    GCCA_score = []
    subj_pair = divide_pair(data_tphate_list_pre_)
    for temp in subj_pair:
        r = canoncorr(data_tphate_list_pre_[temp[0]], data_tphate_list_pre_[temp[1]], fullReturn=False)
        GCCA_score.append(r)
        pow_comp = []
        for comp_i in range(r.shape[0]):
            pow_ , _ = stat_power(r[comp_i], sample_size=data_tphate_list_pre_[temp[0]].shape[0])
            pow_comp.append(pow_)
        power.append(pow_comp)
    GCCA_score_stage.append(np.array(GCCA_score))
    power_stage.append(np.array(power))
    # gcca = GCCA(latent_dimensions=20)
    # gcca.fit(data_tphate_list_pre_)
    # GCCA_score_subj_roi.append(gcca.average_pairwise_correlations(data_tphate_list_pre_))

    # unaligned
    coef_pair = []
    subj_pair = divide_pair(data_tphate_list_pre_)
    for temp in subj_pair:
        # r = canoncorr(data_tphate_list_pre_[temp[0]], data_tphate_list_pre_[temp[1]], fullReturn=False)
        coef = []
        for dim in range(data_tphate_list_pre_[temp[0]].shape[-1]):
            pearson_r = stats.pearsonr(data_tphate_list_pre_[temp[0]][:,dim],
                                       data_tphate_list_pre_[temp[1]][:,dim]).statistic
            coef.append(pearson_r)
        coef_pair.append(np.array(coef))
    coef_stage.append(np.array(coef_pair))

    # post
    time_min = min([data_tphate.shape[0] for data_tphate in data_tphate_list_post])
    data_tphate_list_post_ = [data_tphate[:time_min,:] for data_tphate in data_tphate_list_post]

    # aligned
    power = []
    GCCA_score = []
    subj_pair = divide_pair(data_tphate_list_post_)
    for temp in subj_pair:
        r = canoncorr(data_tphate_list_post_[temp[0]], data_tphate_list_post_[temp[1]], fullReturn=False)
        GCCA_score.append(r)
        pow_comp = []
        for comp_i in range(r.shape[0]):
            pow_, _ = stat_power(r[comp_i], sample_size=data_tphate_list_pre_[temp[0]].shape[0])
            pow_comp.append(pow_)
        power.append(pow_comp)
    GCCA_score_stage.append(np.array(GCCA_score))
    power_stage.append(np.array(power))
    GCCA_score_subj_roi.append(GCCA_score_stage)
    power_subj_roi.append(power_stage)
    # gcca = GCCA(latent_dimensions=20)
    # gcca.fit(data_tphate_list_post_)
    # GCCA_score_subj_roi.append(gcca.average_pairwise_correlations(data_tphate_list_post_))

    # unaligned
    coef_pair = []
    subj_pair = divide_pair(data_tphate_list_post_)
    for temp in subj_pair:
        coef = []
        for dim in range(data_tphate_list_post_[temp[0]].shape[-1]):
            pearson_r = stats.pearsonr(data_tphate_list_post_[temp[0]][:, dim],
                                       data_tphate_list_post_[temp[1]][:, dim]).statistic
            coef.append(pearson_r)
        coef_pair.append(np.array(coef))
    coef_stage.append(np.array(coef_pair))
    coef_subj_roi.append(coef_stage)

# with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/CCA_power_'+Paradigm+'_'+freqb+'.pkl', 'wb') as f:
#     pickle.dump(power_subj_roi, f)

GCCA_roi_diff = []
for i in range(len(GCCA_score_subj_roi)):
    temp = np.array(GCCA_score_subj_roi[i])
    diff = temp[0] - temp[1]
    GCCA_roi_diff.append(diff)

# visualization
# aligned
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
for ii in range(2):
    fig,ax = plt.subplots(ncols=1)
    for i in range(len(GCCA_score_subj_roi)):
        Y = GCCA_score_subj_roi[i][ii]
        shaded_errorbar(ax, np.arange(1,Y.shape[1]+1), Y.T,label=ROIs_label[i])
        # plt.plot(np.array(VAR[i]).T, label=subj_list)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('Neural Modes', fontdict={'size':15})
    ax.set_ylabel('Canonical Correlation', fontdict={'size':15})
    ax.set_xticks(np.arange(2, 21, 2))
    ax.set_title(Paradigm+'-'+freqb+'-'+train_stage[ii], fontdict={'size':15})
    ax.set_ylim([0,0.9])
    fig.tight_layout()
    plt.show()
    fig.savefig('F:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'cross_subj_CCA_' + Paradigm + '_' + freqb + '_'
                + train_stage[ii] +'.eps', format='eps', dpi=1000)

# unaligned
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
for ii in range(2):
    fig,ax = plt.subplots(ncols=1)
    for i in range(len(coef_subj_roi)):
        Y = coef_subj_roi[i][ii]
        shaded_errorbar(ax, np.arange(1,Y.shape[1]+1), Y.T,label=ROIs_label[i])
        # plt.plot(np.array(VAR[i]).T, label=subj_list)
    ax.legend(fontsize=10)
    ax.set_xlabel('Neural Modes', fontdict={'size':15})
    ax.set_ylabel('Correlation', fontdict={'size':15})
    ax.set_xticks(np.arange(2, 21, 2))
    ax.set_title(Paradigm+'-'+freqb+'-'+train_stage[ii], fontdict={'size':15})
    ax.set_ylim([0,0.9])
    fig.tight_layout()
    plt.show()
    fig.savefig('F:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'cross_subj_CCA_' + Paradigm + '_' + freqb + '_'
                + train_stage[ii] +'.eps', format='eps', dpi=1000)


# fig, ax = plt.subplots(ncols=1)
# for i in range(len(CCA_score_roi)):
#     Y = np.array(CCA_score_roi[i])
#     shaded_errorbar(ax, np.arange(1, Y.shape[1] + 1), Y.T, label=ROIs_label[i])
#     # plt.plot(np.array(VAR[i]).T, label=subj_list)
# ax.legend(fontsize=10)
# ax.set_xlabel('Canonical Components', fontdict={'size': 15})
# ax.set_ylabel('Differnece of Canonical Correlation', fontdict={'size': 15})
# ax.set_xticks(np.arange(2, 21, 2))
# ax.set_title(Paradigm + '-' + freqb, fontdict={'size': 15})
# fig.tight_layout()
# plt.show()
# fig.savefig('F:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'structureDiff_' + Paradigm + '_' + freqb + '.eps',
#             format='eps', dpi=1000)

# statistics test - overall
LABELs = []
p_values = []
effectSize_list = []
power_crossSubj = []
for roi_num in range(len(GCCA_score_subj_roi)):
    temp_pre = np.mean(GCCA_score_subj_roi[roi_num][0],-1)
    temp_post = np.mean(GCCA_score_subj_roi[roi_num][1],-1)
    s, p = stats.wilcoxon(np.reshape(temp_pre,-1), np.reshape(temp_post,-1))
    p_values.append(p)
    if p < 0.05:
        LABELs.append(str(ROIs_label[roi_num])+' *')
    else:
        LABELs.append(str(ROIs_label[roi_num]))

    effectSize_list.append(cohens_d(np.reshape(temp_pre,-1), np.reshape(temp_post,-1)))
    pow_, _ = stat_power(effectSize_list[roi_num], sample_size=np.reshape(temp_pre, -1).shape[0])
    power_crossSubj.append(pow_)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\crossSubjSim/crossSubjSimilarityDiff_cohensd_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in effectSize_list:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\crossSubjSim/crossSubjSimilarityDiff_power_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in power_crossSubj:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\crossSubjSim/crossSubjSimilarityDiff_pvalue_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in p_values:
        f.write(f"{item}\n")


# statistics test - box graph (top 4 average)
# LABELs = []
# p_values = []
# for roi_num in range(len(GCCA_score_subj_roi)):
#     temp_pre = np.mean(GCCA_score_subj_roi[roi_num][0][:,:pcNum],axis=1)
#     temp_post = np.mean(GCCA_score_subj_roi[roi_num][1][:,:pcNum],axis=1)
#     s, p = stats.wilcoxon(np.reshape(temp_pre,-1), np.reshape(temp_post,-1))
#     p_values.append(p)
#     if p < 0.05:
#         LABELs.append(str(ROIs_label[roi_num])+' *')
#     else:
#         LABELs.append(str(ROIs_label[roi_num]))

fig, ax = plt.subplots(ncols=1)
for i in range(len(GCCA_roi_diff)):
    Y = np.array(GCCA_roi_diff[i])
    # shaded_errorbar(ax, np.arange(1, Y.shape[-1] + 1), Y.T, label=LABELs[i])
    ax.plot(np.mean(Y,axis=0),label=LABELs[i])
    # plt.plot(np.array(VAR[i]).T, label=subj_list)
# ax.legend(bbox_to_anchor=(1.05,0.25), loc=3, borderaxespad=0,fontsize=10)
ax.legend(loc='upper right',fontsize=12)
ax.set_ylim([-0.05,0.12])
ax.set_xlabel('Canonical Components', fontdict={'size': 15})
ax.set_ylabel('Difference of Canonical Correlation', fontdict={'size': 15})
ax.set_xticks(np.arange(2, 21, 2))
ax.set_title(Paradigm + '-' + freqb, fontdict={'size': 15})
ax.tick_params(labelsize=12)
fig.tight_layout()
plt.show()
save_path = 'G:\CUHK_Intern\RESULTS/figure\Multimodality/'
# fig.savefig(save_path+Paradigm+'_'+freqb+'_crossSubjDiff.eps',format='eps',dpi=1000)


# box plot
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
GCCA_score_temp = []
for i in range(len(GCCA_score_subj_roi)):
    GCCA_score_temp.append(np.mean(np.array(GCCA_score_subj_roi[i])[:,:,:pcNum],axis=-1))
GCCA_score_temp = np.array(GCCA_score_temp)

p_values = []
effectSize_list = []
power_crossSubj_top4 = []
for roi_num in range(len(ROIs)):
    temp_pre = GCCA_score_temp[roi_num,0,:]
    temp_post = GCCA_score_temp[roi_num,1,:]
    s, p = stats.wilcoxon(np.reshape(temp_pre,-1), np.reshape(temp_post,-1))
    p_values.append(p)

    effectSize_list.append(cohens_d(temp_pre, temp_post))
    pow_, _ = stat_power(effectSize_list[roi_num], sample_size=temp_pre.shape[0])
    power_crossSubj_top4.append(pow_)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\crossSubjSim/crossSubjSimilarityDiff_cohensd_'+Paradigm+'_'+freqb+'_top4.txt', 'w') as f:
    for item in effectSize_list:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize\crossSubjSim/crossSubjSimilarityDiff_power_'+Paradigm+'_'+freqb+'_top4.txt', 'w') as f:
    for item in power_crossSubj_top4:
        f.write(f"{item}\n")

data_list = []
# Iterate over stages and ROIs to populate the list
for stage in range(GCCA_score_temp.shape[1]):
    for roi in range(GCCA_score_temp.shape[0]):
        for sample in range(GCCA_score_temp.shape[2]):
            data_list.append([train_stage[stage], ROIs_label[roi], GCCA_score_temp[roi, stage, sample]])

# Convert the list into a DataFrame
df = pd.DataFrame(data_list, columns=['Stage', 'ROI', 'Value'])

# Create a boxplot
fig = plt.figure(figsize=(9, 6))
sns.boxplot(x='ROI', y='Value', hue='Stage', data=df, dodge=True,gap=0.3, palette=['#5F97C6','#F09496'],fill=False)
plt.title('Alteration of Cross-subject Stability',fontsize=15)
plt.xlabel('Regions of Interest',fontsize=15)
plt.ylabel('Averaged Canonical Correlations',fontsize=15)
plt.legend(title='Stage',fontsize=15)
plt.show()
# fig.savefig('F:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'crossSubjDiff_' + Paradigm + '_' + freqb + '.eps',
#             format='eps', dpi=1000)



        # # the consistency across trials could also be compared
        # GCCA_scores_stage = []
        # # pre
        # gcca = GCCA(latent_dimensions=10)
        # data_list = [data_tphate for data_tphate in data_tphate_pre]
        # gcca.fit(data_list)
        # GCCA_scores_stage.append(gcca.average_pairwise_correlations(data_list))
    #     # align across trials
    #     # data_tphate_align_pre = [data_tphate @ gcca.weights_[i] for i, data_tphate in enumerate(data_list)]
    #     data_tphate_align_pre = []
    #     for i in range(len(data_list)):
    #         U, s, Vh = svd(gcca.weights_[i], full_matrices=False, compute_uv=True, overwrite_a=False,
    #                        check_finite=False)
    #         data_tphate_align_pre.append(data_list[i] @ U @ Vh)
    #     data_tphate_reshape_pre = np.reshape(np.array(data_tphate_align_pre), (-1, np.array(data_tphate_align_pre).shape[-1]))
    #     data_tphate_list_pre.append(data_tphate_reshape_pre)
    #
        # # post
        # gcca = GCCA(latent_dimensions=10)
        # data_list = [data_tphate for data_tphate in data_tphate_post]
        # gcca.fit(data_list)
        # GCCA_scores_stage.append(gcca.average_pairwise_correlations(data_list))
        # GCCA_score_trial_subj.append(GCCA_scores_stage)
    #     # align across trials
    #     # data_tphate_align_post = [data_tphate @ gcca.weights_[i] for i, data_tphate in enumerate(data_list)]
    #     data_tphate_align_post = []
    #     for i in range(len(data_list)):
    #         U, s, Vh = svd(gcca.weights_[i], full_matrices=False, compute_uv=True, overwrite_a=False,
    #                        check_finite=False)
    #         data_tphate_align_post.append(data_list[i] @ U @ Vh)
    #     data_tphate_reshape_post = np.reshape(np.array(data_tphate_align_post),
    #                                          (-1, np.array(data_tphate_align_post).shape[-1]))
    #     data_tphate_list_post.append(data_tphate_reshape_post)
    #

    # GCCA_score_trial_roi.append(GCCA_score_trial_subj)







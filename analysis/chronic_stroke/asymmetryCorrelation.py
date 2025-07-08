import numpy as np
from mat73 import loadmat
# from scipy.io import loadmat
from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from scipy import stats
import pickle

# hemisphere = [[9,10],[55,56],[57,58],[125,126]]
hemisphere = [[1,2],[19,20],[59,60],[61,62]]
train_stage = ['pre','post']
Paradigm = 'rest'
freqb = 'theta'

# subj_list = os.listdir(load_path)
subj_list = ['kmt','wws','nsk','nwc','ock','wsc','wwf']
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

            trial_min = min(data_tphate_r.shape[0], data_tphate_l.shape[0])
            rank = min(min(np.linalg.matrix_rank(data_tphate_r)), min(np.linalg.matrix_rank(data_tphate_l)))
            data_tphate_r = data_tphate_r[:trial_min, :, :rank]
            data_tphate_l = data_tphate_l[:trial_min, :, :rank]

            # remove redundant voxels
            # pca = PCA(n_components=30, svd_solver='full')
            # mom_decom_r_pca = pca.fit_transform(mom_decom_r.T)
            #
            # pca = PCA(n_components=30, svd_solver='full')
            # mom_decom_l_pca = pca.fit_transform(mom_decom_l.T)

            # dimensional reduction
            # tphate_op = tphate.TPHATE(n_components=10, knn=7, knn_dist="cosine", mds_dist="cosine")
            # data_tphate_decom_r = tphate_op.fit_transform(mom_decom_r_pca)
            #
            # tphate_op = tphate.TPHATE(n_components=10, knn=7, knn_dist="cosine", mds_dist="cosine")
            # data_tphate_decom_l = tphate_op.fit_transform(mom_decom_l_pca)

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

    rank_min_ = min(data_stage_list_r[0][0].shape[-1],data_stage_list_r[1][0].shape[-1])
    for stage_num in range(2):
        CCA_score_stage = []
        for j in range(len(data_stage_list_l[stage_num])):
            r1 = canoncorr(data_stage_list_r[stage_num][j][:,:rank_min_], data_stage_list_l[stage_num][j][:,:rank_min_], fullReturn=False)
            CCA_score_stage.append(r1)
        CCA_score_roi.append(CCA_score_stage)

    CCA_score.append(CCA_score_roi)


hemisphere_name = ROIs_label = [['PreCG.L','PreCG.R'],['SMA.L','SMA.R'],['SPG.L','SPG.R'],['IPL.L','IPL.R']]
fig,ax = plt.subplots(ncols=1)
p_values = []
effectSize_asym = []
power_asym = []
for hemi_i in range(len(hemisphere)):
    temp = np.array(CCA_score[hemi_i])
    CCA_score_diff = temp[0] - temp[1]
    s, p = stats.wilcoxon(np.mean(temp[0], -1), np.mean(temp[1], -1))
    p_values.append(p)
    if p < 0.05: LABEL = str(hemisphere_name[hemi_i][0])+' - '+str(hemisphere_name[hemi_i][1])+' *'
    else: LABEL = str(hemisphere_name[hemi_i][0])+'-'+str(hemisphere_name[hemi_i][1])
    effectSize_asym.append(cohens_d(np.mean(temp[0], -1), np.mean(temp[1], -1)))
    pow_, _ = stat_power(effectSize_asym[hemi_i], sample_size=np.mean(temp[0], -1).shape[0])
    power_asym.append(pow_)
    # shaded_errorbar(ax, np.arange(1,CCA_score_diff.shape[-1]+1), CCA_score_diff.T,label=LABEL)
    # ax.errorbar(np.arange(1,CCA_score_diff.shape[-1]+1), np.mean(CCA_score_diff,0), yerr=np.std(CCA_score_diff,0),
    #             label=LABEL,fmt='o-', elinewidth=2, capsize=4)
    ax.plot(np.mean(CCA_score_diff, 0), label=LABEL)
    ax.set_ylim([-0.15, 0.2])
    ax.set_xticks(np.arange(2, 21, 2))
    # plt.plot(np.array(VAR[i]).T, label=subj_list)
ax.legend(fontsize=12)
plt.ylabel('Alternation of Canonical Correlation', fontdict={'size':15})
plt.title(Paradigm+'-'+freqb, fontdict={'size':15})
plt.xlabel('Canonical Components', fontdict={'size': 15})
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.show()
# fig.savefig('F:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'AllAsymmetryDiff_' + Paradigm + '_' + freqb + '.eps',
#             format='eps', dpi=1000)

# visualization
hemisphere_name = ROIs_label = [['PreCG.L','PreCG.R'],['SMA.L','SMA.R'],['SPG.L','SPG.R'],['IPL.L','IPL.R']]
fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(6,8))
axs = [ax1,ax2,ax3,ax4]
for hemi_i in range(len(hemisphere)):
    temp = np.array(CCA_score[hemi_i])
    CCA_score_diff = temp[0] - temp[1]
    s, p = stats.wilcoxon(np.mean(temp[0], -1), np.mean(temp[1], -1))
    if p < 0.05: LABEL = str(hemisphere_name[hemi_i][0])+' - '+str(hemisphere_name[hemi_i][1])+' *'
    else: LABEL = str(hemisphere_name[hemi_i][0])+'-'+str(hemisphere_name[hemi_i][1])
    ax = axs[hemi_i]
    # shaded_errorbar(ax, np.arange(1,CCA_score_diff.shape[-1]+1), CCA_score_diff.T,label=LABEL)
    ax.errorbar(np.arange(1,CCA_score_diff.shape[-1]+1), np.mean(CCA_score_diff,0), yerr=np.std(CCA_score_diff,0),
                label=LABEL,fmt='o-', elinewidth=2, capsize=4)
    ax.set_ylim([-0.3, 0.15])
    ax.set_xticks(np.arange(2, 21, 2))
    # plt.plot(np.array(VAR[i]).T, label=subj_list)
fig.legend(loc='center right',fontsize=15)
# plt.ylabel('Alternation of Asymmetry', fontdict={'size':15})
plt.suptitle(Paradigm+'-'+freqb, fontdict={'size':15})
plt.xlabel('Canonical Components', fontdict={'size': 15})
plt.tight_layout()
plt.show()
# fig.savefig('F:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'asymmetryDiff_' + Paradigm + '_' + freqb + '.eps',
#             format='eps', dpi=1000)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/asymmetry/asymmetryDiff_pvalues_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in p_values:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/asymmetry/asymmetryDiff_cohensd_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in effectSize_asym:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/asymmetry/asymmetryDiff_power_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in power_asym:
        f.write(f"{item}\n")
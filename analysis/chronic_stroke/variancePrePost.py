import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from sklearn.decomposition import PCA
import os
from utils import *
from scipy import stats
import pickle

def get_data_mat_var(data_list,n_components):
    model = PCA(n_components=n_components,svd_solver='full')
    rates = np.concatenate(data_list, axis=1)
    rates_model = model.fit(rates.T)
    data_pca = [rates_model.transform(s.T) for s in data_list]

    return data_pca, rates_model.explained_variance_ratio_

# trial_num = 100
ROIs = [1,2,19,20,59,60,61,62]
Paradigm = 'AO1'
freqb = 'alpha'
train_stage = ['pre','post']
threshold = 1 # 0 - 1
pcNum = 4

# load data
load_path = 'G:/CUHK_intern/RESULTS/Multimodality/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt','wws','nsk','nwc','wsc','ock','wwf']

VAR_stage = []
for trainStage in train_stage:
    VAR = []
    for roi in ROIs:
        roi_var = []
        for subj in subj_list:
            data_path = load_path+trainStage+'/'+Paradigm+'/'+subj+'/trial/'
            mom_voxel_list = []

            trial_num = 0
            for file in os.listdir(data_path):
                if file.endswith('.mat'):
                    trial_num += 1

            # trial_num = 26 # for resting state

            if roi % 2 == 0:
                for num in range(1,trial_num+1):
                    # mom_decom = loadmat(load_path+subj+'/'+'trial/'+str(roi)+'/'+subj+'_'+Paradigm+'_'+trainStage+'_decompose_'+
                    #                     freqb+'_'+str(num)+'_l.mat')['mom_decom']
                    mom_voxel = loadmat(data_path + str(roi) + '/' + subj + '_' + Paradigm + '_'
                                        + trainStage + '_voxel_' + str(num) + '_l.mat')['momint_1']
                    # filtering
                    data_filter = eeg_bp_filter(mom_voxel[:, :200], fs=100, freqb=freqb)
                    mom_voxel_list.append(data_filter)
                    del mom_voxel, data_filter
            else:
                for num in range(1,trial_num+1):
                    # mom_decom = loadmat(load_path+subj+'/'+'trial/'+str(roi)+'/'+subj+'_'+Paradigm+'_'+trainStage+'_decompose_'+
                    #                     freqb+'_'+str(num)+'_r.mat')['mom_decom']
                    mom_voxel = loadmat(data_path + str(roi) + '/' + subj + '_' + Paradigm + '_'
                                        + trainStage + '_voxel_' + str(num) + '_r.mat')['momint_1']
                    # filtering
                    data_filter = eeg_bp_filter(mom_voxel[:, :200], fs=100, freqb=freqb)
                    mom_voxel_list.append(data_filter)
                    del mom_voxel, data_filter

            # thresholding and smoothing
            mom_temp = np.concatenate(mom_voxel_list, 1)
            for thres in range(int(np.mean(np.abs(mom_temp), 1).min()),
                               int(np.mean(np.abs(mom_temp), 1).max())):
                voxels_idx = np.mean(np.abs(mom_temp), 1) >= thres
                percent = np.sum(voxels_idx) / mom_temp.shape[0]
                if percent <= threshold:
                    mom_avg_list = []
                    for i, mom_voxel in enumerate(mom_voxel_list):
                        mom_avg_list.append(smooth_average(mom_voxel[voxels_idx, :], 3, 3))  # 30 ms windowing
                    break
            # smoothing
            win = norm_gauss_window(0.03, 0.05)
            mom_smooth_list = [smooth_data(mom_avg_list[i].T, win=win, backend='convolve1d')[10:40, :].T for i
                               in
                               range(len(mom_avg_list))]

            data_pca, var_ratio = get_data_mat_var(mom_smooth_list, 20)

            roi_var.append(var_ratio)
        VAR.append(roi_var)
    VAR_stage.append(VAR)

VAR_stage = np.array(VAR_stage)
VAR_diff = VAR_stage[0,:,:,:] - VAR_stage[1,:,:,:]

ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
# statistics test
LABELs = []
p_values = []
effectSize_var = []
power_var = []
p_pc1 = []
effectSize_pc1 = []
power_pc1 = []
for roi_num in range(VAR_stage.shape[1]):
    s, p = stats.wilcoxon(np.mean(VAR_stage[0, roi_num, :, :],-1), np.mean(VAR_stage[1, roi_num, :, :],-1))
    p_values.append(p)
    if p < 0.05:
        LABELs.append(str(ROIs_label[roi_num])+' *')
    else:
        LABELs.append(str(ROIs_label[roi_num]))

    s, p = stats.wilcoxon(np.reshape(VAR_stage[0, roi_num, :, 0], -1), np.reshape(VAR_stage[1, roi_num, :, 0], -1))
    p_pc1.append(p)

    effectSize_var.append(cohens_d(np.mean(VAR_stage[0, roi_num, :, :],-1), np.mean(VAR_stage[1, roi_num, :, :],-1)))
    pow_, _ = stat_power(effectSize_var[roi_num], sample_size=np.mean(VAR_stage[0, roi_num, :, :],-1).shape[0])
    power_var.append(pow_)

    effectSize_pc1.append(cohens_d(np.reshape(VAR_stage[0, roi_num, :, 0], -1), np.reshape(VAR_stage[1, roi_num, :, 0], -1)))
    pow_, _ = stat_power(effectSize_pc1[roi_num], sample_size=np.reshape(VAR_stage[0, roi_num, :, 0], -1).shape[0])
    power_pc1.append(pow_)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/variance/varianceDiff_cohensd_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in effectSize_var:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/variance/varianceDiff_power_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in power_var:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/variance/varianceDiff_pvalue_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in p_values:
        f.write(f"{item}\n")

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/variance/variancePC1Diff_pvalues_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in p_pc1:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/variance/variancePC1Diff_cohensd_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in effectSize_pc1:
        f.write(f"{item}\n")
with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/variance/variancePC1Diff_power_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in power_pc1:
        f.write(f"{item}\n")

# visualization
fig,ax = plt.subplots(ncols=1)
for i in range(VAR_diff.shape[0]):
    # shaded_errorbar(ax, np.arange(1,21), np.array(VAR_diff[i]).T,label=LABELs[i])
    ax.plot(np.mean(VAR_diff[i,:,:10],axis=0),label=LABELs[i])
    # plt.plot(np.array(VAR[i]).T, label=subj_list)
# ax.legend(bbox_to_anchor=(1.05,0.25), loc=3, borderaxespad=0,fontsize=10)
ax.legend(loc='upper right',fontsize=12)
ax.set_xlabel('Principal Components', fontdict={'size':15})
ax.set_ylabel('Difference of Explained Variances', fontdict={'size':15})
ax.set_title(Paradigm+'-'+freqb, fontdict={'size':15})
ax.set_xticks(np.arange(2,11,2))
ax.tick_params(labelsize=12)
ax.set_ylim([-0.15,0.25])
fig.tight_layout()
plt.show()

save_path = 'F:\CUHK_Intern\RESULTS/figure\Multimodality/'
# fig.savefig(save_path+Paradigm+'_'+freqb+'_varDiff.eps',format='eps',dpi=1000)

# proportion bar plot
# for ii in range(len(train_stage)):
#     var_temp = VAR_stage[ii,:,:,:]
#     ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
#     components = ['PC1','PC2','PC3','PC4','Others']
#     color = np.array([(219,49,36),(252,140,90),(255,223,146),(230,241,243),(144,190,224),(75,116,178)])/255
#     # color = np.array([(144,201,230),(33,158,188),(2,48,71),(255,183,3),(251,132,2)])/255
#     # color = np.array([(231,56,71),(240,250,239),(168,218,219),(69,123,157),(29,53,87)])/255
#     var_temp = np.concatenate((var_temp[:,:,:pcNum],np.sum(var_temp[:,:,pcNum:],axis=-1,keepdims=True)),axis=-1)
#     var_temp_avg = np.mean(var_temp,axis=1)
#     fig = plt.figure(figsize=(8,4),dpi=300)
#     bottom_vals = np.zeros(len(ROIs_label))
#     for i in range(var_temp_avg.shape[-1]):
#         plt.bar(ROIs_label, var_temp_avg[:,i], width=0.8,bottom=bottom_vals,
#                 label= components[i], color=color[i], edgecolor='grey')
#         bottom_vals += var_temp_avg[:,i]
#     plt.ylim([0,1.01])
#     # plt.show()
#     plt.tick_params(axis='x',length=0)
#     plt.xlabel('Regions of Interest', fontsize=15)
#     plt.ylabel('Explained Variance(%)', fontsize=15)
#     plt.grid(axis='y',alpha=0.5,ls='--')
#     plt.legend(frameon=False, bbox_to_anchor=(1.01,1), fontsize=12)
#     plt.tight_layout()
#     plt.show()
#     save_path = 'F:\CUHK_Intern\RESULTS/figure\Multimodality/'
#     fig.savefig(save_path + 'variance_proportion_bar_'+ train_stage[ii] +'.eps', dpi=1000,format='eps')



var_pre = VAR_stage[0,:,:,:]
var_post = VAR_stage[1,:,:,:]
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
components = ['PC1','PC2','PC3','PC4','Others']
# color = np.array([(219,49,36),(252,140,90),(255,223,146),(230,241,243),(144,190,224),(75,116,178)])/255
color = np.array([(75,116,178),(144,190,224),(230,241,243),(255,223,146),(252,140,90),(219,49,36)])/255
# color = np.array([(144,201,230),(33,158,188),(2,48,71),(255,183,3),(251,132,2)])/255
# color = np.array([(231,56,71),(240,250,239),(168,218,219),(69,123,157),(29,53,87)])/255
# color = np.array([(90,180,229),(154,208,240),(236,206,223),(217,155,187),(206,121,167)])/255
var_pre = np.concatenate((var_pre[:,:,:pcNum],np.sum(var_pre[:,:,pcNum:],axis=-1,keepdims=True)),axis=-1)
var_pre_avg = np.mean(var_pre,axis=1)
var_post = np.concatenate((var_post[:,:,:pcNum],np.sum(var_post[:,:,pcNum:],axis=-1,keepdims=True)),axis=-1)
var_post_avg = np.mean(var_post,axis=1)

fig, ax = plt.subplots(ncols=1,figsize=(10,5),dpi=300)
bottom_vals_pre = np.zeros(len(ROIs_label))
bottom_vals_post = np.zeros(len(ROIs_label))
x = np.arange(0, len(ROIs))+1
width = 0.45
for i in range(var_pre_avg.shape[-1]):
    rects1 = ax.bar(x - width/2 - 0.01, var_pre_avg[:,i], width=width,bottom=bottom_vals_pre,
                    label=components[i], color=color[i], edgecolor='none')
    bottom_vals_pre += var_pre_avg[:,i]
    rects2 = ax.bar(x + width/2 + 0.01, var_post_avg[:, i], width=width, bottom=bottom_vals_post,
                    color=color[i], edgecolor='none')
    bottom_vals_post += var_post_avg[:, i]
ax.set_ylim([0,1.01])
# plt.show()
ax.set_xticks(x)
ax.set_xticklabels(ROIs_label,fontsize=12)
ax.set_xlabel('Regions of Interest', fontsize=15)
ax.set_ylabel('Explained Variance(%)', fontsize=15)
plt.grid(axis='y',alpha=0.5,ls='--')
plt.legend(frameon=False, bbox_to_anchor=(1.01,1), fontsize=12)
plt.tight_layout()
plt.show()
save_path = 'F:\CUHK_Intern\RESULTS/figure\Multimodality/'
# fig.savefig(save_path + 'variance_diff_proportion_bar.eps', dpi=1000,format='eps')
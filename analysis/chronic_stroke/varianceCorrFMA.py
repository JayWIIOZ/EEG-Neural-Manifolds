'''
For looking at this, it makes no sense on comparing neural modes before and after directly because neural mode may change,
better to CCA align first and then find out the corresponding neural modes before and after.
'''

import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from sklearn.decomposition import PCA
import os
from Code.utils import *
from scipy import stats
import pandas as pd
from scipy.stats import pearsonr, spearmanr

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
load_path = 'F:/CUHK_intern/RESULTS/Multimodality/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt','wws','nsk','nwc','wsc','ock','wwf']

# load FMA scores
df = pd.read_excel('F:/CUHK_Intern/subj_info.xlsx')
subj_fma = [[df.name],[df['FMA_Pre']],[df['FMA_Post']]]
# subj_list = df['name'].tolist()
subj_list = ['kmt','wws','nsk','nwc','ock','wsc','wwf']

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

# obtain FMA scores
fma_scores_all = []
for i in range(2):
    fma_scores = []
    for subj in subj_list:
        fma_scores.append(subj_fma[i + 1][0][subj_fma[0][0] == subj].values)
    fma_scores_all.append(fma_scores)

fma_scores_all_temp = np.reshape(fma_scores_all,-1)

for roiNum in range(len(ROIs)):
    var_temp = np.reshape(VAR_stage[:,roiNum,:,:],(-1,VAR_stage.shape[-1]))
    spearmanr(var_temp,fma_scores_all_temp)


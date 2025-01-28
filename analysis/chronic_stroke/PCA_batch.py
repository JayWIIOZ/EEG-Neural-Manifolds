import numpy as np
from mat73 import loadmat
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import random
from scipy import signal
from Code.utils import *

# directly using voxel lead to low cross-subject stability

def smooth_average(mom_decom, window_len, step):
    '''
    Smooth the data using a moving average.
    data: matrix of voxels*samples
    window_len: window length
    step: step of the moving window
    '''
    window_len = window_len
    step = step
    mom_avg = []
    counting = 0
    while (counting <= mom_decom.shape[1] - 1):
        if counting == 0:
            mom_avg.append(np.mean(mom_decom[:, :counting + window_len // 2 + 1], 1))
            counting += step
        elif counting == mom_decom.shape[1] - 1:
            mom_avg.append(np.mean(mom_decom[:, counting - window_len // 2:], 1))
            break
        else:
            mom_avg.append(np.mean(mom_decom[:, counting - window_len // 2:counting + window_len // 2 + 1], 1))
            counting += step
    mom_avg = np.array(mom_avg).T

    return mom_avg

def down_sampling(data_list):
    voxel_list = [x.shape[0] for x in data_list]
    target_voxel = min(voxel_list)
    data_cut_list = []
    for i, voxel_num in enumerate(voxel_list):
        idx = random.sample(range(voxel_num),target_voxel)
        idx = np.sort(idx)
        data_cut_list.append(data_list[i][idx,:])

    return data_cut_list

def get_data_mat(data_list,n_components):
    model = PCA(n_components=n_components,svd_solver='full')
    rates = np.concatenate(data_list, axis=1)
    rates_model = model.fit(rates.T)
    data_pca = [rates_model.transform(s.T) for s in data_list]

    return data_pca, np.cumsum(rates_model.explained_variance_ratio_)

def eeg_bp_filter(data, fs, freqb='all', order=4):
    '''
    data: matrix of voxels*samples
    fs: sampling frequency
    freqb: frequency band for filtering, 'all' means no filtering, 'delta' [1 4], 'theta' [4 8], 'alpha' [8 12],
    'beta' [13 30], 'gamma' [30 40]
    order: order of filtering
    '''
    freqBand = {'delta': [1, 4],
                'theta': [4, 8],
                'alpha': [8, 12],
                'beta': [12, 30],
                'gamma': [30, 40]}
    if freqb not in freqBand:
        data_filter = data
    else:
        lf, hf = freqBand[freqb]
        wn1 = 2*lf/fs
        wn2 = 2*hf/fs
        b, a = signal.butter(order, [wn1, wn2], 'bandpass')
        data_filter = signal.filtfilt(b, a, data, axis=0)

    return data_filter


ROIs = [1,2,19,20,59,60,61,62]
# ROIs = [1]
Paradigm = 'AO1'
freqb = 'alpha' # 'all', 'delta', 'theta', 'alpha', 'beta', 'gamma'
trainStage = 'post'
threshold = 1 # refers to remaining 20% of voxels

# load data
load_path = 'F:/CUHK_intern/RESULTS/Multimodality/'+trainStage+'/'+Paradigm+'/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt','wws','nsk','nwc','ock','wsc','wwf','lkk','pcy']
# subj_list = ['lkk','pcy']

for roi in ROIs:
    for subj in subj_list:
        save_path = ('F:/CUHK_intern/RESULTS/Multimodality/'
                     + trainStage + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/')
        decom_tphate_trial = []
        mom_decom_list = []

        # for AO1, count trial num
        trial_num = 0
        for file in os.listdir(load_path+subj+'/trial'):
            if file.endswith('.mat'):
                trial_num += 1

        # for rest, use default num
        # trial_num = 26

        if roi % 2 == 0:
            for num in range(1, trial_num + 1):
                # mom_decom = loadmat(load_path + subj + '/' + 'trial/' + str(roi) + '/' + subj + '_' + Paradigm + '_'
                #                     + trainStage + '_decompose_' + str(num) + '_l.mat')['mom_decom']
                mom_decom = loadmat(load_path + subj + '/' + 'trial/' + str(roi) + '/' + subj + '_' + Paradigm + '_'
                                    + trainStage + '_voxel_' + str(num) + '_l.mat')['momint_1']
                # filtering
                data_filter = eeg_bp_filter(mom_decom[:, 200:400], fs=100, freqb=freqb)
                mom_decom_list.append(data_filter)

                del mom_decom, data_filter
        else:
            for num in range(1, trial_num + 1):
                mom_decom = loadmat(load_path + subj + '/' + 'trial/' + str(roi) + '/' + subj + '_' + Paradigm + '_'
                                    + trainStage + '_voxel_' + str(num) + '_r.mat')['momint_1']
                # filtering
                data_filter = eeg_bp_filter(mom_decom[:, 200:400], fs=100, freqb=freqb)
                mom_decom_list.append(data_filter)

                del mom_decom, data_filter

        # thresholding
        # mom_avg_list = []
        # for i, mom_decom in enumerate(mom_decom_list):
        #     for thres in range(10000, int(np.sum(np.abs(mom_decom),1).max()), 500):
        #         voxels_idx = np.sum(np.abs(mom_decom), 1) >= thres
        #         percent = np.sum(voxels_idx)/mom_decom.shape[0]
        #         if percent <= threshold:
        #             mom_avg_list.append(smooth_average(mom_decom[voxels_idx, :],3,3))
        #             # mom_avg_list.append(mom_decom[voxels_idx, :])
        #             break

        # thresholding and smoothing
        mom_temp = np.concatenate(mom_decom_list,1)
        for thres in range(int(np.mean(np.abs(mom_temp),1).min()),int(np.mean(np.abs(mom_temp),1).max())):
            voxels_idx = np.mean(np.abs(mom_temp), 1) >= thres
            percent = np.sum(voxels_idx) / mom_temp.shape[0]
            if percent <= threshold:
                mom_avg_list = []
                for i, mom_decom in enumerate(mom_decom_list):
                    mom_avg_list.append(smooth_average(mom_decom[voxels_idx, :], 3, 3))
                break
        # smoothing
        win = norm_gauss_window(0.03, 0.05)
        mom_smooth_list = [smooth_data(mom_avg_list[i].T,win=win,backend='convolve1d')[10:40,:].T for i in range(len(mom_avg_list))]

        data_pca, var_ratio = get_data_mat(mom_smooth_list, 20)




        # data_cut_list = down_sampling(mom_avg_list)
        # win = norm_gauss_window(0.03, 0.05)
        # mom_smooth_list = [smooth_data(data_cut_list[i].T, win=win, backend='convolve1d')[10:40, :].T for i in
        #                    range(len(data_cut_list))]
        # data_pca, var_ratio = get_data_mat(mom_smooth_list,20)
        np.save(save_path+subj+'_'+Paradigm+'_'+trainStage+'_pca_trial_mid_'+freqb+'.npy',data_pca)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(3,4))
# ax = plt.axes(projection='3d', fc='None')
plt.matshow(np.concatenate(mom_smooth_list,axis=1),cmap='jet')
# ax.plot(data_pca[2][:,0],data_pca[2][:,1],data_pca[2][:,2])
plt.show()



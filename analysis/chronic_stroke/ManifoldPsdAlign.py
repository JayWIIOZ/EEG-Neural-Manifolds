import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
import os
from scipy import signal
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from utils import *

'''
for next step, better to see if the changes on psd also reflects in manifolds
'''

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x)*np.linalg.norm(y)
    return num/denom

class Config:
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
    train_stage = ['pre','post']
    Paradigm = 'AO1'
    freqb = 'alpha'
    subj_list = ['kmt', 'ock']
    fs = 100
    nperseg = 25
    noverlap = 24
    freqBand = {'delta':[1,4],
                'theta':[4,8],
                'alpha':[8,12],
                'beta':[12,30],
                'gamma':[30,40]}
    mat_path = 'chronic_stroke/'
    pca_path = 'chronic_stroke/pca_data/'
    save_path = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/manifold_psd'
    trial_min = 13

os.makedirs(Config.save_path, exist_ok=True)

CCA_score = []
for trainStage in Config.train_stage:
    # load data
    CCA_score_stage = []
    load_path = os.path.join(Config.mat_path, trainStage, Config.Paradigm)

    for roi in Config.ROIs:
        CCA_score_roi = []

        for subj in Config.subj_list:
            data_voxel_list = []
            CCA_score_subj = []
            pca_path = os.path.join(Config.pca_path, trainStage, Config.Paradigm, subj, str(roi), f'{subj}_pca_trial_{Config.freqb}.npy')
            data_pca = np.load(pca_path)
            if roi % 2 == 0:
                for num in range(1, Config.trial_min + 1):
                    mat_path = os.path.join(load_path, subj, str(roi), f'{subj}_{Config.Paradigm}_{trainStage}_voxel_{num}_l.mat')
                    data_voxel = loadmat(mat_path)['momint_1']
                    data_voxel_list.append(data_voxel[:,:200])
            else:
                for num in range(1, Config.trial_min + 1):
                    mat_path = os.path.join(load_path, subj, str(roi), f'{subj}_{Config.Paradigm}_{trainStage}_voxel_{num}_r.mat')
                    data_voxel = loadmat(mat_path)['momint_1']
                    data_voxel_list.append(data_voxel[:,:200])

            voxel_psd_list = []
            for trial_ii in range(Config.trial_min):
                f, t, Zxx = signal.stft(data_voxel_list[trial_ii], Config.fs, nperseg=Config.nperseg, noverlap=Config.noverlap, scaling='psd')
                voxel_psd = np.mean(Zxx[:, np.logical_and(f >= Config.freqBand[Config.freqb][0], f <= Config.freqBand[Config.freqb][1]), :].real, axis=1)
                voxel_psd_ = smooth_average(voxel_psd,3,3)[:, 10:40]
                voxel_psd_list.append(voxel_psd_)

            psd_pca, var_ratio, _ = get_data_mat(voxel_psd_list, 20)
            rank_min = min(min(np.linalg.matrix_rank(psd_pca)), min(np.linalg.matrix_rank(data_pca)))
            psd_pca_reshape = np.reshape(np.array(psd_pca)[:, :, :rank_min], (-1, rank_min))
            data_pca_reshape = np.reshape(data_pca[:Config.trial_min, :, :rank_min], (-1, rank_min))
            r = canoncorr(data_pca_reshape, psd_pca_reshape, fullReturn=False)
            CCA_score_subj.append(r)
            CCA_score_roi.append(CCA_score_subj)

            # decom_psd = np.mean(np.mean(Zxx[:, np.logical_and(f >= freqBand[freqb][0], f <= freqBand[freqb][1]), :], axis=1), axis=0)
            # CCA_score_dim = []
            # for dim in range(data_tphate.shape[1]):
            #     CCA_score_dim.append(cosine_similarity(data_tphate[:, dim], decom_psd.real))
            # CCA_score_roi.append(CCA_score_dim)
            #
            # decom_psd = np.mean(Zxx[:, np.logical_and(f >= freqBand[freqb][0], f <= freqBand[freqb][1]), :],axis=1)
            # pca = PCA(n_components=10, svd_solver='full')
            # psd_pca = pca.fit_transform(decom_psd.real.T)
            # A, B, r, *_ = canoncorr(data_tphate, psd_pca[:200, :], fullReturn=True)
            # CCA_score_roi.append(r)
            #
            # decom_psd = Zxx[:, np.logical_and(f >= freqBand[freqb][0], f <= freqBand[freqb][1]), :]
            # CCA_score_freq = []
            # for freq_num in range(decom_psd.shape[1]):
            #     pca = PCA(n_components=10, svd_solver='full')
            #     psd_pca = pca.fit_transform(decom_psd[:, freq_num, :].real.T)
            #
            #     A, B, r, *_ = canoncorr(data_tphate, psd_pca[:200,:], fullReturn=True)
            #     CCA_score_freq.append(r)
            # CCA_score_roi.append(CCA_score_freq)
        dim_min = min([CCA_score_roi[i][0].shape for i in range(len(CCA_score_roi))])[0]
        CCA_score_stage.append([temp[0][:dim_min] for temp in CCA_score_roi])
    CCA_score.append(CCA_score_stage)

# visualization
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
# Pre
fig1,ax1 = plt.subplots(ncols=1)
for i in range(len(CCA_score[0])):
    Y = np.array(CCA_score[0][i])
    # shaded_errorbar(ax1, np.arange(1, Y.shape[-1]+1), Y.T,label=str(ROIs_label[i]))
    ax1.plot(np.arange(1,np.mean(Y,axis=0).shape[0]+1), np.mean(Y,axis=0), label=str(ROIs_label[i]))
    # plt.plot(np.array(VAR[i]).T, label=subj_list)
ax1.legend(fontsize=10)
ax1.set_xlabel('Dimensions', fontdict={'size':15})
ax1.set_ylabel('Canonical Correlation', fontdict={'size':15})
ax1.set_title(Config.Paradigm+'-'+Config.freqb+'-Pre', fontdict={'size':15})
ax1.set_ylim([0,1.01])
ax1.set_xticks(np.arange(0,19,2))
# ax1.set_xticklabels(np.arange(0,19,2))
fig1.tight_layout()
plt.show()
fig1.savefig(Config.save_path +
             'ManiPsd_'+Config.Paradigm+'_' + Config.freqb + '_Pre.png', format='png', dpi=1000)

# Post
fig2,ax2 = plt.subplots(ncols=1)
for i in range(len(CCA_score[1])):
    Y = np.array(CCA_score[1][i])
    # shaded_errorbar(ax2, np.arange(1, Y.shape[-1]+1), Y.T,label=str(ROIs_label[i]))
    ax2.plot(np.arange(1,np.mean(Y,axis=0).shape[0]+1), np.mean(Y, axis=0), label=str(ROIs_label[i]))
    # plt.plot(np.array(VAR[i]).T, label=subj_list)
ax2.legend(fontsize=10)
ax2.set_xlabel('Dimensions', fontdict={'size':15})
ax2.set_ylabel('Canonical Correlation', fontdict={'size':15})
ax2.set_title(Config.Paradigm+'-'+Config.freqb+'-Post', fontdict={'size':15})
ax2.set_ylim([0,1.01])
ax2.set_xticks(np.arange(0,19,2))
fig2.tight_layout()
plt.show()
fig2.savefig(Config.save_path +
             'ManiPsd_'+Config.Paradigm+'_' + Config.freqb + '_Post.png', format='png', dpi=1000)

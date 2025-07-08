import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from sklearn.decomposition import PCA
import os
from utils import *
from scipy import stats
import pickle
import scipy.io as sio

class Config:
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
    train_stage = ['high','low']
    Paradigm = ['AO','rest']
    freqb = ['beta', 'alpha', 'theta', 'delta']
    ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
    threshold = 1 # 0 - 1
    pcNum = 4
    save_path = 'analysis_result'
    mat_path = '/Users/cizer/Downloads/taoliu/rest_data'
    stroke_data_path = 'stroke_data'
    rest_data_path = 'rest_data'

def get_data_mat_var(data_list,n_components):
    model = PCA(n_components=n_components,svd_solver='full')
    rates = np.concatenate(data_list, axis=1)
    rates_model = model.fit(rates.T)
    data_pca = [rates_model.transform(s.T) for s in data_list]

    return data_pca, rates_model.explained_variance_ratio_

def load_data(cf, paradigm):
    if paradigm == 'AO':
        datapath = 'stroke_data'
    elif paradigm == 'rest':
        datapath = 'rest_data'
    
    high_subj_path = os.path.join(datapath, 'high')
    low_subj_path = os.path.join(datapath, 'low')
    high_subj = os.listdir(high_subj_path)
    low_subj = os.listdir(low_subj_path)
    subj_list = []
    subj_list.append(high_subj)
    subj_list.append(low_subj)
    return subj_list


def get_variance(cf, subj_list, freqb):
    VAR_stage = []
    for stage_idx in range(2):
        VAR = []
        for roi in cf.ROIs:
            roi_var = []
            for subj in subj_list[stage_idx]:
                print(f'subj: {subj}, roi: {roi}, freqb: {freqb}')
                path = os.path.join(cf.mat_path, subj, f'{subj}_trial_roi_rest{str(roi)}', subj, 'trial', str(roi))
                mom_voxel_list = []

                if roi % 2 == 0:
                    for num in range(1,27):
                        mom_voxel = sio.loadmat(path+f'/{subj}_voxel_rest{num}_l.mat')['momint_1']
                        data_filter = eeg_bp_filter(mom_voxel[:, :200], fs=100, freqb=freqb)
                        mom_voxel_list.append(data_filter)
                        del mom_voxel, data_filter
                else:
                    for num in range(1,27):
                        mom_voxel = sio.loadmat(path+f'/{subj}_voxel_rest{num}_r.mat')['momint_1']
                        data_filter = eeg_bp_filter(mom_voxel[:, :200], fs=100, freqb=freqb)
                        mom_voxel_list.append(data_filter)
                        del mom_voxel, data_filter

                mom_temp = np.concatenate(mom_voxel_list, 1)
                for thres in range(int(np.mean(np.abs(mom_temp), 1).min()),
                                int(np.mean(np.abs(mom_temp), 1).max())):
                    voxels_idx = np.mean(np.abs(mom_temp), 1) >= thres
                    percent = np.sum(voxels_idx) / mom_temp.shape[0]
                    if percent <= cf.threshold:
                        mom_avg_list = []
                        for i, mom_voxel in enumerate(mom_voxel_list):
                            mom_avg_list.append(smooth_average(mom_voxel[voxels_idx, :], 3, 3))  # 30 ms windowing
                        break

                # smoothing
                win = norm_gauss_window(0.03, 0.05)
                mom_smooth_list = [smooth_data(mom_avg_list[i].T, win=win, backend='convolve1d')[10:40, :].T for i
                                in
                                range(len(mom_avg_list))]

                data_pca, var_ratio = get_data_mat_var(mom_smooth_list, 25)

                roi_var.append(var_ratio)
            VAR.append(roi_var)
        VAR_stage.append(VAR)
    VAR_stage = np.array(VAR_stage)
    VAR_diff = VAR_stage[0,:,:,:] - VAR_stage[1,:,:,:]


    return VAR_stage, VAR_diff

def visualize_result(cf, VAR_stage, VAR_diff, save_path, paradim, freq):
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
            LABELs.append(str(cf.ROIs_label[roi_num])+' *')
        else:
            LABELs.append(str(cf.ROIs_label[roi_num]))

        s, p = stats.wilcoxon(np.reshape(VAR_stage[0, roi_num, :, 0], -1), np.reshape(VAR_stage[1, roi_num, :, 0], -1))
        p_pc1.append(p)

        effectSize_var.append(cohens_d(np.mean(VAR_stage[0, roi_num, :, :],-1), np.mean(VAR_stage[1, roi_num, :, :],-1)))
        pow_, _ = stat_power(effectSize_var[roi_num], sample_size=np.mean(VAR_stage[0, roi_num, :, :],-1).shape[0])
        power_var.append(pow_)

        effectSize_pc1.append(cohens_d(np.reshape(VAR_stage[0, roi_num, :, 0], -1), np.reshape(VAR_stage[1, roi_num, :, 0], -1)))
        pow_, _ = stat_power(effectSize_pc1[roi_num], sample_size=np.reshape(VAR_stage[0, roi_num, :, 0], -1).shape[0])
        power_pc1.append(pow_)

        # 将数据保存到dataframe中
    import pandas as pd
    data = {
        'ROIs': LABELs,
        'p_values': p_values,
        'effectSize_var': effectSize_var,
        'power_var': power_var,
        'p_pc1': p_pc1,
        'effectSize_pc1': effectSize_pc1,
        'power_pc1': power_pc1
    }
    df = pd.DataFrame(data)
    df_savepath = os.path.join(save_path, 'powerAndEffectSize', 'variance')
    os.makedirs(df_savepath, exist_ok=True)
    df.to_csv(os.path.join(df_savepath, f'{paradim}_{freq}_variance_analysis.csv'), index=False)

    fig,ax = plt.subplots(ncols=1)
    for i in range(VAR_diff.shape[0]):
        # shaded_errorbar(ax, np.arange(1,21), np.array(VAR_diff[i]).T,label=LABELs[i])
        ax.plot(np.mean(VAR_diff[i,:,:10],axis=0),label=LABELs[i])
        # plt.plot(np.array(VAR[i]).T, label=subj_list)
    # ax.legend(bbox_to_anchor=(1.05,0.25), loc=3, borderaxespad=0,fontsize=10)
    ax.legend(loc='upper right',fontsize=12)
    ax.set_xlabel('Principal Components', fontdict={'size':15})
    ax.set_ylabel('Difference of Explained Variances', fontdict={'size':15})
    ax.set_title(paradim+'-'+freq, fontdict={'size':15})
    ax.set_xticks(np.arange(2,11,2))
    ax.tick_params(labelsize=12)
    ax.set_ylim([-0.15,0.25])
    fig.tight_layout()
    fig_save_path = os.path.join(df_savepath, 'figure')
    fig.savefig(fig_save_path+f'{paradim}_{freq}_varDiff.png', format='png', dpi=1000)
    fig.savefig(fig_save_path+f'{paradim}_{freq}_varDiff.eps', format='eps', dpi=1000)

    var_pre = VAR_stage[0,:,:,:]
    var_post = VAR_stage[1,:,:,:]
    ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
    components = ['PC1','PC2','PC3','PC4','Others']
    # color = np.array([(219,49,36),(252,140,90),(255,223,146),(230,241,243),(144,190,224),(75,116,178)])/255
    color = np.array([(75,116,178),(144,190,224),(230,241,243),(255,223,146),(252,140,90),(219,49,36)])/255
    # color = np.array([(144,201,230),(33,158,188),(2,48,71),(255,183,3),(251,132,2)])/255
    # color = np.array([(231,56,71),(240,250,239),(168,218,219),(69,123,157),(29,53,87)])/255
    # color = np.array([(90,180,229),(154,208,240),(236,206,223),(217,155,187),(206,121,167)])/255
    var_pre = np.concatenate((var_pre[:,:,:cf.pcNum],np.sum(var_pre[:,:,cf.pcNum:],axis=-1,keepdims=True)),axis=-1)
    var_pre_avg = np.mean(var_pre,axis=1)
    var_post = np.concatenate((var_post[:,:,:cf.pcNum],np.sum(var_post[:,:,cf.pcNum:],axis=-1,keepdims=True)),axis=-1)
    var_post_avg = np.mean(var_post,axis=1)

    fig, ax = plt.subplots(ncols=1,figsize=(10,5),dpi=300)
    bottom_vals_pre = np.zeros(len(ROIs_label))
    bottom_vals_post = np.zeros(len(ROIs_label))
    x = np.arange(0, len(cf.ROIs))+1
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
    fig.savefig(fig_save_path + f'variance_diff_proportion_bar_{paradim}_{freq}.eps', dpi=1000,format='eps')
    fig.savefig(fig_save_path + f'variance_diff_proportion_bar_{paradim}_{freq}.png', dpi=1000, format='png')


def main():
    cf = Config()
    subj_list = load_data(cf, 'rest')
    for f in cf.freqb:
        VAR_stage, VAR_diff = get_variance(cf, subj_list, f)
        visualize_result(cf, VAR_stage, VAR_diff, cf.save_path, cf.Paradigm[1], f)


if __name__ == '__main__':
    main()


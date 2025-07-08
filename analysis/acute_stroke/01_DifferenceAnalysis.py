import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import pandas as pd
import seaborn as sns
import pickle

class Config:
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
    # ROIs = [1]
    train_stage = ['high','low']
    Paradigm = ['AO','rest']
    freqb = ['beta', 'alpha', 'theta', 'delta']
    save_path = 'analysis_result'
    pcNum = 4
    ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']

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


def prepare_data(cf, subj_list, paradigm, freqb):
    if paradigm == 'AO':
        datapath = 'stroke_data'
    elif paradigm == 'rest':
        datapath = 'rest_data'
    power_subj_roi = []
    CCA_score_roi = []
    GCCA_score_subj_roi = []
    coef_subj_roi = []
    for roi in cf.ROIs:
        CCA_score_subj = []
        data_tphate_list = []
        for i in range(len(cf.train_stage)):
            temp_list = []
            for subj in subj_list[i]:
                data_path = os.path.join(datapath, cf.train_stage[i], subj, f'roi_{roi}', f'{subj}_{paradigm}_pca_trial_{freqb}.npy')
                data_tphate = np.load(data_path)
                trial_min = data_tphate.shape[0]
                rank = min(np.linalg.matrix_rank(data_tphate))
                data_tphate = data_tphate[:trial_min, :, :rank]
                data_tphate_reshape = np.reshape(data_tphate, (-1, data_tphate.shape[-1]))
                temp_list.append(data_tphate_reshape)

            data_tphate_list.append(temp_list)

        rank_min = min([data_tmp.shape[1] for data_tmp in data_tphate_list[0]])
        for i in range(len(data_tphate_list[0])):
            data_tphate_list[0][i] = data_tphate_list[0][i][:,:rank_min]
            data_tphate_list[1][i] = data_tphate_list[1][i][:, :rank_min]
        
        for i in range(len(data_tphate_list[0])):
            r1 = canoncorr(data_tphate_list[0][i], data_tphate_list[1][i], fullReturn=False)
            CCA_score_subj.append(r1)

        CCA_score_roi.append(CCA_score_subj)

        # cross subject consistency for high group
        power_stage = []
        GCCA_score_stage = []
        coef_stage = []
        # high group
        time_min = min([data_tphate.shape[0] for data_tphate in data_tphate_list[0]])
        data_tphate_list_high = [data_tphate[:time_min,:] for data_tphate in data_tphate_list[0]]

        # aligned
        power = []
        GCCA_score = []
        subj_pair = divide_pair(data_tphate_list_high)
        for temp in subj_pair:
            r = canoncorr(data_tphate_list_high[temp[0]], data_tphate_list_high[temp[1]], fullReturn=False)
            GCCA_score.append(r)
            pow_comp = []
            for comp_i in range(r.shape[0]):
                pow_ , _ = stat_power(r[comp_i], sample_size=data_tphate_list_high[temp[0]].shape[0])
                pow_comp.append(pow_)
            power.append(pow_comp)
        GCCA_score_stage.append(np.array(GCCA_score))
        power_stage.append(np.array(power))

        # unaligned
        coef_pair = []
        subj_pair = divide_pair(data_tphate_list_high)
        for temp in subj_pair:
            # r = canoncorr(data_tphate_list_pre_[temp[0]], data_tphate_list_pre_[temp[1]], fullReturn=False)
            coef = []
            for dim in range(data_tphate_list_high[temp[0]].shape[-1]):
                pearson_r = stats.pearsonr(data_tphate_list_high[temp[0]][:,dim],
                                           data_tphate_list_high[temp[1]][:,dim]).statistic
                coef.append(pearson_r)
            coef_pair.append(np.array(coef))
        coef_stage.append(np.array(coef_pair))

        # low group
        time_min = min([data_tphate.shape[0] for data_tphate in data_tphate_list[1]])
        data_tphate_list_low = [data_tphate[:time_min,:] for data_tphate in data_tphate_list[1]]
        
        # aligned
        power = []
        GCCA_score = []
        subj_pair = divide_pair(data_tphate_list_low)
        for temp in subj_pair:
            r = canoncorr(data_tphate_list_low[temp[0]], data_tphate_list_low[temp[1]], fullReturn=False)
            GCCA_score.append(r)
            pow_comp = []
            for comp_i in range(r.shape[0]):
                pow_, _ = stat_power(r[comp_i], sample_size=data_tphate_list_low[temp[0]].shape[0])
                pow_comp.append(pow_)
            power.append(pow_comp)
        power_stage.append(np.array(power))
        power_subj_roi.append(power_stage)
        GCCA_score_stage.append(np.array(GCCA_score))
        GCCA_score_subj_roi.append(GCCA_score_stage)

        # unaligned
        coef_pair = []
        subj_pair = divide_pair(data_tphate_list_low)
        for temp in subj_pair:
            coef = []
            for dim in range(data_tphate_list_low[temp[0]].shape[-1]):
                pearson_r = stats.pearsonr(data_tphate_list_low[temp[0]][:, dim],
                                           data_tphate_list_low[temp[1]][:, dim]).statistic
                coef.append(pearson_r)
            coef_pair.append(np.array(coef))
        coef_stage.append(np.array(coef_pair))
        coef_subj_roi.append(coef_stage)

    GCCA_roi_diff = []
    for i in range(len(GCCA_score_subj_roi)):
        temp = np.array(GCCA_score_subj_roi[i])
        diff = temp[0] - temp[1]
        GCCA_roi_diff.append(diff)

    return power_subj_roi, coef_subj_roi, CCA_score_roi, GCCA_score_subj_roi, GCCA_roi_diff



def statistics_visulization(cf, subj_list, save_path, paradigm, freq):
    LABELs = []
    p_values = []
    effectSize_list = []
    power_crossSubj = []
    power_subj_roi, coef_subj_roi, CCA_score_roi, GCCA_score_subj_roi, GCCA_roi_diff = prepare_data(cf, subj_list, paradigm, freq)

    for roi_num in range(len(GCCA_score_subj_roi)):
        temp_high = np.mean(GCCA_score_subj_roi[roi_num][0],-1)
        temp_low = np.mean(GCCA_score_subj_roi[roi_num][1],-1)
        s, p = stats.wilcoxon(np.reshape(temp_high,-1), np.reshape(temp_low,-1))
        p_values.append(p)
        if p < 0.05:
            LABELs.append(str(cf.ROIs_label[roi_num])+' *')
        else:
            LABELs.append(str(cf.ROIs_label[roi_num]))

        effectSize_list.append(cohens_d(np.reshape(temp_high,-1), np.reshape(temp_low,-1)))
        pow_, _ = stat_power(effectSize_list[roi_num], sample_size=np.reshape(temp_high, -1).shape[0])
        power_crossSubj.append(pow_)
    write_list = ['cohensd', 'power', 'pvalue']
    data_list = [effectSize_list, power_crossSubj, p_values]
    data_dict = {
        'ROI': cf.ROIs_label,
        'Cohens_d': effectSize_list,
        'Power': power_crossSubj,
        'P_value': p_values
    }
    df = pd.DataFrame(data_dict)
    path_to_write = os.path.join(save_path, 'powerAndEffectSize', 'crossSubjSim')
    if not os.path.exists(path_to_write):
        os.makedirs(path_to_write)
    csv_path = os.path.join(path_to_write, f'crossSubjSimilarityDiff_{paradigm}_{freq}.csv')
    df.to_csv(csv_path, index=False)
    

    fig, ax = plt.subplots(ncols=1)
    for i in range(len(GCCA_roi_diff)):
        Y = np.array(GCCA_roi_diff[i])
        ax.plot(np.mean(Y,axis=0),label=LABELs[i])

    ax.legend(loc='upper right',fontsize=12)
    ax.set_ylim([-0.05,0.12])
    ax.set_xlabel('Canonical Components', fontdict={'size': 15})
    ax.set_ylabel('Difference of Canonical Correlation', fontdict={'size': 15})
    ax.set_xticks(np.arange(2, 21, 2))
    ax.set_title(paradigm + '-' + freq, fontdict={'size': 15})
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig_save_path = os.path.join(save_path, 'figure', 'DifferenceAnalysis')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    plt.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_diff.png', format = 'png',dpi=1000)
    plt.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_diff.eps', format = 'eps',dpi=1000)
    # box plot
    GCCA_score_temp = []
    for i in range(len(GCCA_score_subj_roi)):
        GCCA_score_temp.append(np.mean(np.array(GCCA_score_subj_roi[i])[:,:,:cf.pcNum],axis=-1))
    GCCA_score_temp = np.array(GCCA_score_temp)

    p_values = []
    effectSize_list = []
    power_crossSubj_top4 = []
    for roi_num in range(len(cf.ROIs)):
        temp_pre = GCCA_score_temp[roi_num,0,:]
        temp_post = GCCA_score_temp[roi_num,1,:]
        s, p = stats.wilcoxon(np.reshape(temp_pre,-1), np.reshape(temp_post,-1))
        p_values.append(p)

        effectSize_list.append(cohens_d(temp_pre, temp_post))
        pow_, _ = stat_power(effectSize_list[roi_num], sample_size=temp_pre.shape[0])
        power_crossSubj_top4.append(pow_)
    
    with open(path_to_write+f'/crossSubjSimilarityDiff_cohensd_{cf.Paradigm}_{cf.freqb}_top4.txt', 'w') as f:
        for item in effectSize_list:
            f.write(f"{item}\n")
    with open(path_to_write+f'/crossSubjSimilarityDiff_power_{cf.Paradigm}_{cf.freqb}_top4.txt', 'w') as f:
        for item in power_crossSubj_top4:
            f.write(f"{item}\n")

    data_list = []
    # Iterate over stages and ROIs to populate the list
    for stage in range(GCCA_score_temp.shape[1]):
        for roi in range(GCCA_score_temp.shape[0]):
            for sample in range(GCCA_score_temp.shape[2]):
                data_list.append([cf.train_stage[stage], cf.ROIs_label[roi], GCCA_score_temp[roi, stage, sample]])

    # Convert the list into a DataFrame
    df = pd.DataFrame(data_list, columns=['Stage', 'ROI', 'Value'])

    # Create the boxplot
    fig = plt.figure(figsize=(9, 6))
    sns.boxplot(x='ROI', y='Value', hue='Stage', data=df, dodge=True,gap=0.3, palette=['#5F97C6','#F09496'],fill=False)
    plt.title('Alteration of Cross-subject Stability',fontsize=15)
    plt.xlabel('Regions of Interest',fontsize=15)
    plt.ylabel('Averaged Canonical Correlations',fontsize=15)
    plt.legend(title='Stage',fontsize=15)
    plt.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_boxplot.png', format='png', dpi=1000)
    plt.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_boxplot.eps', format='eps', dpi=1000)


def cross_subject_cca_visualize(cf, coef_subj_roi, GCCA_score_subj_roi, paradigm, freq, save_path):
    fig_save_path = os.path.join(save_path, 'figure', 'crossSubjCCA')
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)


    for ii in range(2):
        fig,ax = plt.subplots(ncols=1)
        for i in range(len(GCCA_score_subj_roi)):
            Y = GCCA_score_subj_roi[i][ii]
            shaded_errorbar(ax, np.arange(1,Y.shape[1]+1), Y.T,label=cf.ROIs_label[i])
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.set_xlabel('Neural Modes', fontdict={'size':15})
        ax.set_ylabel('Canonical Correlation', fontdict={'size':15})
        ax.set_xticks(np.arange(2, 21, 2))
        ax.set_title(paradigm+'-'+freq+'-'+cf.train_stage[ii], fontdict={'size':15})
        ax.set_ylim([0,0.9])
        fig.tight_layout()
        fig.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_cca_{cf.train_stage[ii]}.png', format='png',dpi=1000)
        fig.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_cca_{cf.train_stage[ii]}.eps', format='eps',dpi=1000)

    for ii in range(2):
        fig,ax = plt.subplots(ncols=1)
        for i in range(len(coef_subj_roi)):
            Y = coef_subj_roi[i][ii]
            shaded_errorbar(ax, np.arange(1,Y.shape[1]+1), Y.T,label=cf.ROIs_label[i])
            # plt.plot(np.array(VAR[i]).T, label=subj_list)
        ax.legend(fontsize=10)
        ax.set_xlabel('Neural Modes', fontdict={'size':15})
        ax.set_ylabel('Correlation', fontdict={'size':15})
        ax.set_xticks(np.arange(2, 21, 2))
        ax.set_title(paradigm+'-'+freq+'-'+cf.train_stage[ii], fontdict={'size':15})
        ax.set_ylim([0,0.9])
        fig.tight_layout()
        fig.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_coef_{cf.train_stage[ii]}.png', format='png',dpi=1000)
        fig.savefig(fig_save_path+f'/{paradigm}_{freq}_cross_subj_coef_{cf.train_stage[ii]}.eps', format='eps',dpi=1000)


def main():
    cf = Config()
    paradigm = cf.Paradigm
    freq = cf.freqb

    save_path = 'analysis_result'
    for p in paradigm:
        subj_list = load_data(cf, p)
        for f in freq:
            power_subj_roi, coef_subj_roi, CCA_score_roi, GCCA_score_subj_roi, GCCA_roi_diff = prepare_data(cf, subj_list, p, f)
            cross_subject_cca_visualize(cf, coef_subj_roi,GCCA_score_subj_roi, p, f, save_path)


if __name__ == "__main__":
    main()

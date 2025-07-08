import numpy as np
from mat73 import loadmat
# from scipy.io import loadmat
from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from scipy import stats
import pickle
import pandas as pd
from scipy.stats import pearsonr


class Config:
    hemisphere = [[1,2],[19,20],[59,60],[61,62]]
    train_stage = ['high','low']
    Paradigm = ['AO', 'rest']
    freqb = ['alpha','beta', 'theta','delta']
    stroke_data_path = 'stroke_data'
    rest_data_path = 'rest_data'

    save_path = 'analysis_result'

def load_data(cf):
    datapath = 'stroke_data'
    high_subj_path = os.path.join(datapath, 'high')
    low_subj_path = os.path.join(datapath, 'low')
    high_subj = os.listdir(high_subj_path)
    low_subj = os.listdir(low_subj_path)
    subj_list = []
    subj_list.append(high_subj)
    subj_list.append(low_subj)
    return subj_list

def combine_scores(CCA_score, nihss_scores):
    """
    将 CCA 分数与 NIHSS 分数对应组合
    
    参数:
        CCA_score: 形状为 (4, 2, 15) 的数组
        nihss_scores: 形状为 (2, 15) 的列表或数组
        
    返回:
        包含所有组合数据的 DataFrame
    """

    data = []
    
    # 遍历所有组合
    for brain_idx in range(4):        # 遍历 4 个脑区
        for stage_idx in range(2):    # 遍历 2 个阶段
            for subject_idx in range(15):  # 遍历 15 个受试者
                # 获取对应分数
                cca_val = CCA_score[brain_idx][stage_idx][subject_idx]
                nihss_val = nihss_scores[stage_idx][subject_idx]
                
                # 添加记录
                data.append({
                    'brain_region': brain_idx,
                    'stage': stage_idx,
                    'subject': subject_idx,
                    'cca_score': cca_val,
                    'nihss_score': nihss_val
                })
    
    # 创建 DataFrame
    df = pd.DataFrame(data)
    return df

def calculate_pairwise_differences(df):
    """
    对每组受试者进行两两配对，计算CCA和NIHSS的差值
    参数:
        df: 包含脑区、阶段、CCA分数和NIHSS分数的DataFrame
    返回:
        包含所有配对的DataFrame
    """
    # 1. 分离两组受试者的数据
    # 假设阶段0是第一组（高分组），阶段1是第二组（低分组）
    group1 = df[df['stage'] == 0].copy()
    group1 = group1.rename(columns={
        'subject': 'subject1',
        'cca_score': 'cca_score1',
        'nihss_score': 'nihss_score1'
    })
    
    group2 = df[df['stage'] == 1].copy()
    group2 = group2.rename(columns={
        'subject': 'subject2',
        'cca_score': 'cca_score2',
        'nihss_score': 'nihss_score2'
    })
    
    # 2. 创建配对模板（每组脑区的所有可能配对）
    group1['key'] = 1
    group2['key'] = 1
    pairs = pd.merge(group1, group2, on=['brain_region', 'key'])
    pairs = pairs.drop(columns='key')
    
    # 3. 计算差值
    pairs['cca_diff'] = pairs['cca_score1'] - pairs['cca_score2']
    pairs['nihss_diff'] = pairs['nihss_score1'] - pairs['nihss_score2']
    
    # 4. 添加配对ID
    pairs['pair_id'] = pairs.index
    
    return pairs



def analysis(cf, paradigm, freq):
    if paradigm == 'AO':
        datapath = cf.stroke_data_path
    elif paradigm == 'rest':
        datapath = cf.rest_data_path
    
    high_subj_path = os.path.join(datapath, 'high')
    low_subj_path = os.path.join(datapath, 'low')
    high_subj = os.listdir(high_subj_path)
    low_subj = os.listdir(low_subj_path)
    subj_list = []
    subj_list.append(high_subj)
    subj_list.append(low_subj)

    # Get NIHSS scores
    csv_path = 'participants_copy.csv'
    nihss_scores_all = pd.read_csv(csv_path)
    nihss_scores = []
    for stage_idx, stage_subjs in enumerate(subj_list):
        stage_scores = []
        for subj in stage_subjs:
            score = nihss_scores_all[nihss_scores_all['Participant_ID'] == subj]['NIHSS'].values[0]
            stage_scores.append(score)
        nihss_scores.append(stage_scores)

    CCA_score = []
    for roi_num in range(len(cf.hemisphere)):
        CCA_score_roi = []
        data_stage_list_r = []
        data_stage_list_l = []
        for trainStage in range(len(cf.train_stage)):
            data_tphate_list_r = []
            data_tphate_list_l = []
            for subj in subj_list[trainStage]:
                data_path_r = os.path.join(datapath, cf.train_stage[trainStage], subj, f'roi_{str(cf.hemisphere[roi_num][0])}', f'{subj}_{paradigm}_pca_trial_{freq}.npy')
                data_tphate_r = np.load(data_path_r)
                data_path_l = os.path.join(datapath, cf.train_stage[trainStage], subj, f'roi_{str(cf.hemisphere[roi_num][1])}', f'{subj}_{paradigm}_pca_trial_{freq}.npy')
                data_tphate_l = np.load(data_path_l)

                trial_min = min(data_tphate_r.shape[0], data_tphate_l.shape[0])
                rank = min(min(np.linalg.matrix_rank(data_tphate_r)), min(np.linalg.matrix_rank(data_tphate_l)))
                data_tphate_r = data_tphate_r[:trial_min, :, :rank]
                data_tphate_l = data_tphate_l[:trial_min, :, :rank]

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

    return CCA_score, nihss_scores

def correlationPreprocess(CCA_score, nihss_scores):

    CCA_SCORE = []
    for i in range(4):
        cca_roi_score = CCA_score[i]
        cca_roi_score = np.array(cca_roi_score)
        avg_cca_roi_score = np.mean(cca_roi_score, axis=-1)
        print(avg_cca_roi_score.shape)
        CCA_SCORE.append(avg_cca_roi_score)
    CCA_score = np.array(CCA_SCORE)
    nihss_scores = np.array(nihss_scores)
    return CCA_score, nihss_scores


def traversalPair(cf, CCA_score):
    """
    Traversal all pairs of CCA scores.
    :param CCA_score: CCA scores for each hemisphere.
    :return: List of pairs of CCA scores.
    CCA_score[roi_num][0]: High NIHSS
    CCA_score[roi_num][1]: Low NIHSS
    """
    CCA_SCORE = []
    for i in range(4):
        cca_roi_score = CCA_score[i]
        cca_roi_score = np.array(cca_roi_score)
        avg_cca_roi_score = np.mean(cca_roi_score, axis=-1)
        print(avg_cca_roi_score.shape)
        CCA_SCORE.append(avg_cca_roi_score)
    CCA_SCORE = np.array(CCA_SCORE)
    def divide_pair(group_score):
        pair_pairs = []
        for high in group_score[0]:
            for low in group_score[1]:
                pair_pairs.append((high, low))
        return pair_pairs
    CCA_pairs_score = []
    for roi_num in range(len(cf.hemisphere)):
        pair_pairs = divide_pair(CCA_SCORE[roi_num])
        CCA_pairs_score.append(pair_pairs)
    
    return CCA_pairs_score

    
def analyze_correlations(pairs, save_path, freq, paradigm):
    """
    分析每个脑区CCA差值与NIHSS差值的相关性
    参数:
        pairs: 包含所有配对的DataFrame
    返回:
        包含相关系数和p值的DataFrame
    """
    # 存储结果
    results = []
    
    # 遍历所有脑区
    unique_brain_regions = pairs['brain_region'].unique()
    for brain_region in unique_brain_regions:
        # 提取当前脑区的所有配对
        brain_pairs = pairs[pairs['brain_region'] == brain_region]
        
        
        corr, p_value = pearsonr(brain_pairs['cca_diff'], brain_pairs['nihss_diff'])
        pow_, _ = stat_power(corr, sample_size=len(brain_pairs['nihss_diff']))
        
        results.append({
            'brain_region': brain_region,
            'correlation': corr,
            'p_value': p_value,
            'power': pow_,
            'n_pairs': len(brain_pairs)
        })
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 添加显著性标记
    results_df['significant'] = results_df['p_value'] < 0.05

    # 将result_df保存为CSV文件
    save = os.path.join(save_path, 'additionalResult', 'powerAndEffectSize','asymmetryCorrNIHSS')
    if not os.path.exists(save):
        os.makedirs(save)
    results_df.to_csv(os.path.join(save_path, f'reslut_{paradigm}_{freq}.csv'), index=False)
    
    return results_df

# def visualization(cf, pairs, paradigm, freqb, save_path):
#     hemisphere_name = ROIs_label = [['PreCG.L','PreCG.R'],['SMA.L','SMA.R'],['SPG.L','SPG.R'],['IPL.L','IPL.R']]
#     fig,ax = plt.subplots(ncols=1)
#     p_values = []
#     effectSize_asym = []
#     power_asym = []
#     for hemi_i in range(len(cf.hemisphere)):
#         region_data = pairs[pairs['brain_region'] == hemi_i]
#         s, p = stats.wilcoxon(region_data['cca_score1'], region_data['cca_score2'])
#         p_values.append(p)
#         if p < 0.05: LABEL = str(hemisphere_name[hemi_i][0])+' - '+str(hemisphere_name[hemi_i][1])+' *'
#         else: LABEL = str(hemisphere_name[hemi_i][0])+'-'+str(hemisphere_name[hemi_i][1])
#         effectSize_asym.append(cohens_d(region_data['cca_score1'], region_data['cca_score2']))
#         pow_, _ = stat_power(effectSize_asym[hemi_i], sample_size=len(region_data['cca_score1']))
#         power_asym.append(pow_)

#         ax.plot(region_data['cca_diff'], label=LABEL)
#         # ax.set_ylim()
#         # ax.set_xticks(np.arange(2, 21, 2))
#     ax.legend(fontsize=12)
#     plt.ylabel('Alternation of Canonical Correlation', fontdict={'size':15})
#     plt.title(paradigm+'-'+freqb, fontdict={'size':15})
#     plt.xlabel('Canonical Components', fontdict={'size': 15})
#     ax.tick_params(labelsize=12)
#     plt.tight_layout()
#     figsave_path = os.path.join(save_path, 'figure', 'AsymmetryAnalysis', 'AllAsymmetryDiff',paradigm)
#     if not os.path.exists(figsave_path):
#         os.makedirs(figsave_path)
#     fig.savefig(figsave_path+f'/AllAsymmetryDiff_{paradigm}_{freqb}.png', format='png', dpi=1000)
#     # fig.savefig(figsave_path+f'/AllAsymmetryDiff_{paradigm}_{freqb}.eps', format='eps', dpi=1000)
    

#     # hemisphere_name = ROIs_label = [['PreCG.L','PreCG.R'],['SMA.L','SMA.R'],['SPG.L','SPG.R'],['IPL.L','IPL.R']]
#     # fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(6,8))
#     # axs = [ax1,ax2,ax3,ax4]
#     # for hemi_i in range(len(cf.hemisphere)):
#     #     region_data = pairs[pairs['brain_region'] == hemi_i]
#     #     s, p = stats.wilcoxon(region_data['cca_score1'], region_data['cca_score2'])
#     #     if p < 0.05: LABEL = str(hemisphere_name[hemi_i][0])+' - '+str(hemisphere_name[hemi_i][1])+' *'
#     #     else: LABEL = str(hemisphere_name[hemi_i][0])+'-'+str(hemisphere_name[hemi_i][1])
#     #     ax = axs[hemi_i]
#     #     cca_diff_values = region_data['cca_diff'].values
#     #     x_range = np.arange(len(cca_diff_values))
#     #     mean_diff = np.mean(cca_diff_values)
#     #     std_diff = np.std(cca_diff_values)
#     #     ax.errorbar(x=cca_diff_values, 
#     #                 y=mean_diff,
#     #                 yerr=std_diff,
#     #                 label=LABEL,fmt='o-', elinewidth=2, capsize=4)
#     #     # ax.set_ylim([-0.3, 0.15])
#     #     # ax.set_xticks(np.arange(2, 21, 2))
#     # fig.legend(loc='center right',fontsize=15)
#     # # plt.ylabel('Alternation of Asymmetry', fontdict={'size':15})
#     # plt.suptitle(paradigm+'-'+freqb, fontsize=15)
#     # plt.xlabel('Canonical Components', fontsize=15)
#     # plt.tight_layout()
#     # # plt.savefig(figsave_path+f'/asymmetryDiff_{paradigm}_{freqb}.eps', format='eps', dpi=1000)
#     # plt.savefig(figsave_path+f'/asymmetryDiff_{paradigm}_{freqb}.png', format='png', dpi=1000)

#     results_dict = {
#         'ROI': [f"{h[0]}-{h[1]}" for h in hemisphere_name],
#         'p_value': p_values,
#         'cohens_d': effectSize_asym,
#         'power': power_asym
#     }

#     results_df = pd.DataFrame(results_dict)
    
#     # 保存路径
#     file_save_path = os.path.join(save_path, 'powerAndEffectSize', 'asymmetry')
#     if not os.path.exists(file_save_path):
#         os.makedirs(file_save_path)
        
#     # 保存CSV文件
#     csv_path = os.path.join(file_save_path, f'asymmetryDiff_stats_{paradigm}_{freqb}.csv')
#     results_df.to_csv(csv_path, index=False)




def visualize(cf, CCA_score, paradigm, freqb, save_path):
    hemisphere_name = ROIs_label = [['PreCG.L','PreCG.R'],['SMA.L','SMA.R'],['SPG.L','SPG.R'],['IPL.L','IPL.R']]
    fig,ax = plt.subplots(ncols=1)
    p_values = []
    effectSize_asym = []
    power_asym = []
    for hemi_i in range(len(cf.hemisphere)):
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
    plt.title(paradigm+'-'+freqb, fontdict={'size':15})
    plt.xlabel('Canonical Components', fontdict={'size': 15})
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    figsave_path = os.path.join(save_path, 'figure', 'AsymmetryAnalysis', 'AllAsymmetryDiff',paradigm)
    if not os.path.exists(figsave_path):
        os.makedirs(figsave_path)
    fig.savefig(figsave_path+f'/AllAsymmetryDiff_{paradigm}_{freqb}.png', format='png', dpi=1000)
    fig.savefig(figsave_path+f'/AllAsymmetryDiff_{paradigm}_{freqb}.eps', format='eps', dpi=1000)
    # fig.savefig('F:\CUHK_Intern\RESULTS/figure\Multimodality/' + 'AllAsymmetryDiff_' + Paradigm + '_' + freqb + '.eps',
    #             format='eps', dpi=1000)

    results_dict = {
        'ROI': [f"{h[0]}-{h[1]}" for h in hemisphere_name],
        'p_value': p_values,
        'cohens_d': effectSize_asym,
        'power': power_asym
    }
    
    # 创建DataFrame
    results_df = pd.DataFrame(results_dict)
    
    # 保存路径
    file_save_path = os.path.join(save_path, 'powerAndEffectSize', 'asymmetry')
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)
        
    # 保存CSV文件
    csv_path = os.path.join(file_save_path, f'asymmetryDiff_stats_{paradigm}_{freqb}.csv')
    results_df.to_csv(csv_path, index=False)

    # file_save_path = os.path.join(save_path, 'powerAndEffectSize', 'asymmetry')
    # if not os.path.exists(file_save_path):
    #     os.makedirs(file_save_path)
    # with open(file_save_path+f'/asymmetryDiff_pvalues_{paradigm}_{freqb}.txt', 'w') as f:
    #     for item in p_values:
    #         f.write(f"{item}\n")
    # with open(file_save_path+f'/asymmetryDiff_cohensd_{paradigm}_{freqb}.txt', 'w') as f:
    #     for item in effectSize_asym:
    #         f.write(f"{item}\n")
    # with open(file_save_path+f'/asymmetryDiff_power_{paradigm}_{freqb}.txt', 'w') as f:
    #     for item in power_asym:
    #         f.write(f"{item}\n")


def nihss_correlation(cf, nihss_scores):
    s, p = stats.wilcoxon(nihss_scores[0], nihss_scores[1])
    return s, p



def main():
    cf = Config()
    save_path = cf.save_path
    for f in cf.freqb:
        for p in cf.Paradigm:
            CCA_score, nihss_score = analysis(cf, p, f)
            # visualize(cf, CCA_score, p, f, save_path)
            # CCA_score, nihss_score = correlationPreprocess(CCA_score, nihss_score)
            # df = combine_scores(CCA_score, nihss_score)
            # pairs = calculate_pairwise_differences(df)
            # visualization(cf, pairs, save_path, f, p)
            # # result_df = analyze_correlations(pairs, save_path, f, p)
            s,p = nihss_correlation(cf, nihss_score)
            print(s,p)

if __name__ == "__main__":
    main()
    

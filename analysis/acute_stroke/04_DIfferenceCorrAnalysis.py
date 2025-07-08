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


class Config:
    ROIs = [1, 2, 19, 20, 59, 60, 61, 62]
    # ROIs = [1]
    train_stage = ['high','low']
    Paradigm = ['AO','rest']
    freqb = ['beta', 'alpha', 'theta', 'delta']
    stroke_data_path = 'stroke_data'
    rest_data_path = 'rest_data'
    save_path = 'analysis_result'

def get_scores(cf, subj_list):
    csv_path = 'participants_copy.csv'
    nihss_scores_all = pd.read_csv(csv_path)
    nihss_scores = []
    for stage_idx, stage_subjs in enumerate(subj_list):
        stage_scores = []
        for subj in stage_subjs:
            score = nihss_scores_all[nihss_scores_all['Participant_ID'] == subj]['NIHSS'].values[0]
            stage_scores.append(score)
        nihss_scores.append(stage_scores)

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

def get_scores(cf, subj_list):
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
    return nihss_scores

def analyze_correlations(cf, subj_list, nihss_scores, save_path, freq, paradigm, roi):
    if paradigm == 'AO':
        datapath = 'stroke_data'
    elif paradigm == 'rest':
        datapath = 'rest_data'

    nihss_scores = get_scores(cf, subj_list)
    nihss_scores = np.array(nihss_scores)

    data_dict = {
        'subj_id': [],
        'brain_region': [],  # 脑区索引
        'stage': [],  # 'high' 或 'low'
        'nihss_score': [],  # NIHSS分数
        'data': []  # 对应的数据矩阵
    }

    dis_roi = []
    GCCA_score_subj_roi = []
    data_aligned_roi = []
    
    dis_subj = []
    data_tphate_list_high = []
    data_tphate_list_low = []
    for idx, subj in enumerate(subj_list[0]):
        data_path_high = os.path.join(datapath, 'high', subj, f'roi_{roi}')
        data_tphate_high = np.load(os.path.join(data_path_high, f'{subj}_{paradigm}_pca_trial_{freq}.npy'))
        data_tphate_list_high.append(data_tphate_high)
        data_dict['subj_id'].append(subj)
        data_dict['brain_region'].append(roi)
        data_dict['stage'].append('high')
        data_dict['nihss_score'].append(nihss_scores[0][idx])
        data_dict['data'].append(data_tphate_high)
    for idx, subj in enumerate(subj_list[1]):
        data_path_low = os.path.join(datapath, 'low', subj, f'roi_{roi}')
        data_tphate_low = np.load(os.path.join(data_path_low, f'{subj}_{paradigm}_pca_trial_{freq}.npy'))
        data_tphate_list_low.append(data_tphate_low)
        data_dict['subj_id'].append(subj)
        data_dict['brain_region'].append(roi)
        data_dict['stage'].append('low')
        data_dict['nihss_score'].append(nihss_scores[1][idx])
        data_dict['data'].append(data_tphate_low)
    
    time_len = data_tphate_list_high[0].shape[1]
    trial_min = min([data.shape[0] for data in data_tphate_list_high + data_tphate_list_low])
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data_dict)

    for i, data in enumerate(data_tphate_list_low):
        rank_list = np.linalg.matrix_rank(data)
        rank_low = min(rank_list)
    for i, data in enumerate(data_tphate_list_high):
        rank_high_list = np.linalg.matrix_rank(data)
        rank_high = min(rank_high_list)
    rank = min(rank_low, rank_high)
    # rank = min(min(np.linalg.matrix_rank(data) for data in data_tphate_list_high),
    #            min(np.linalg.matrix_rank(data) for data in data_tphate_list_low))
    data_tphate_list_high = [data[:trial_min, :, :rank] for data in data_tphate_list_high]
    data_tphate_list_low = [data[:trial_min, :, :rank] for data in data_tphate_list_low]
    data_tphate_reshape_high = [np.reshape(data, (-1, data.shape[-1])) for data in data_tphate_list_high]
    data_tphate_reshape_low = [np.reshape(data, (-1, data.shape[-1])) for data in data_tphate_list_low]
    data_aligned = []

    df_high = df[df['stage'] == 'high'].copy()
    df_low = df[df['stage'] == 'low'].copy()

    # 替换data列
    df_high['data'] = data_tphate_reshape_high
    df_low['data'] = data_tphate_reshape_low

    # 将15个高分组和15个低分组一一配对，共225个组合
    df_high['key'] = 1
    df_low['key'] = 1
    pairs = pd.merge(df_high, df_low, on=['brain_region', 'key'])
    pairs = pairs.drop(columns='key')
    pairs['nihss_diff'] = pairs['nihss_score_x'] - pairs['nihss_score_y']
    pairs['pair_id'] = range(len(pairs))
    mani_diff = []
    for i in range(len(pairs)):
        A1, B1, r1, *_ = canoncorr(pairs.iloc[i]['data_x'], pairs.iloc[i]['data_y'], fullReturn=True)
        U1, s1, Vh1 = np.linalg.svd(A1, full_matrices=False, compute_uv=True)
        U2, s2, Vh2 = np.linalg.svd(B1, full_matrices=False, compute_uv=True)
        temp_pre = np.reshape(pairs.iloc[i]['data_x'] @ U1 @ Vh1, (-1, time_len, r1.shape[-1]))
        temp_post = np.reshape(pairs.iloc[i]['data_y'] @ U2 @ Vh2, (-1, time_len, r1.shape[-1]))
        data_aligned.append([np.mean(temp_pre, 0), np.mean(temp_post, 0)])
        # mani_diff = np.array([grassmann_distance(np.mean(temp_pre,0)[ii,:], np.mean(temp_post,0)[ii,:]) for ii in range(temp_pre.shape[1])])
        # 将mani_diff保存到pairs中
        temp_diff = grassmann_distance(temp_pre, temp_post)
        temp_diff = np.mean(temp_diff, axis=1)
        mani_diff.append(temp_diff)
    pairs['mani_diff'] = mani_diff


    return pairs

def show_result(cf, data_aligned_roi, paradim, freq):

    results = []

    for roi_num in range(len(data_aligned_roi)):
        temp = data_aligned_roi[roi_num]
        res = spearmanr(temp['mani_diff'], temp['nihss_diff'])
        pow_, _ = stat_power(res.correlation + 1e-13, len(temp['nihss_diff']))

        results.append({
            'ROI': cf.ROIs[roi_num],
            'Correlation': res.correlation,
            'P-value': res.pvalue,
            'Power': pow_
        })

    result_df = pd.DataFrame(results)   

    save = os.path.join(cf.save_path, 'additionalResult', 'powerAndEffectSize','differenceCorrNIHSS')
    if not os.path.exists(save):
        os.makedirs(save)
    result_df.to_csv(os.path.join(save, f'{paradim}_{freq}_correlation_results.csv'), index=False)
    
def main():
    cf = Config()
    paradigms = cf.Paradigm
    freqs = cf.freqb
    s = load_data(cf, paradigms[0])
    nihss_scores = get_scores(cf, s)
    for p in paradigms:
        subj_list = load_data(cf, p)
        for f in freqs:
            data_aligned_roi = []
            for roi in cf.ROIs:
                pairs = analyze_correlations(cf, subj_list, nihss_scores, cf.save_path, f, p, roi)
                data_aligned_roi.append(pairs)
            # show_result(cf, data_aligned_roi, p, f)

   

if __name__ == "__main__":
    main()

    

import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from sklearn.decomposition import PCA
import os
from utils import *
from scipy import stats
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from scipy.linalg import subspace_angles
import pickle

def get_data_var_weight(data_list,n_components):
    model = PCA(n_components=n_components,svd_solver='full')
    rates = np.concatenate(data_list, axis=1)
    rates_model = model.fit(rates.T)
    data_pca = [rates_model.transform(s.T) for s in data_list]

    return rates_model.explained_variance_ratio_, rates_model.components_

# trial_num = 100
ROIs = [1,2,19,20,59,60,61,62]
Paradigm = 'rest'
freqb = 'alpha'
train_stage = ['pre','post']
threshold = 1 # 0 - 1
pcNum = 4

# load data
load_path = 'G:/CUHK_intern/RESULTS/Multimodality/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt','wws','nsk','nwc','wsc','ock','wwf']

# load FMA scores
df = pd.read_excel('G:/CUHK_Intern/subj_info.xlsx')
subj_fma = [[df.name],[df['FMA_Pre']],[df['FMA_Post']]]
# subj_list = df['name'].tolist()
subj_list = ['kmt','wws','nsk','nwc','ock','wsc','wwf']

weight_stage = []
var_stage = []
for trainStage in train_stage:
    weight = []
    var_roi = []
    for roi in ROIs:
        roi_weight = []
        var_subj = []
        for subj in subj_list:
            data_path = load_path+trainStage+'/'+Paradigm+'/'+subj+'/trial/'
            mom_voxel_list = []

            # trial_num = 0
            # for file in os.listdir(data_path):
            #     if file.endswith('.mat'):
            #         trial_num += 1

            trial_num = 26 # for resting state

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

            var_pca, pcWeights = get_data_var_weight(mom_smooth_list, 20)

            roi_weight.append(pcWeights)
            var_subj.append(var_pca)
        weight.append(roi_weight)
        var_roi.append(var_subj)
    weight_stage.append(weight)
    var_stage.append(var_roi)

# similarity between PCA weights before and after BCI training (this can help identify if the neural modes are similar)
pairs_roi = []
scores_roi = []
for roi_num in range(len(ROIs)):
    scores_subj = []
    pairs_subj = []
    for subj_num in range(len(subj_list)):
        if subj_num == 4:
            continue
        else:
            matched_pairs, similarity_scores, _ = compare_pc_weights(weight_stage[0][roi_num][subj_num], weight_stage[1][roi_num][subj_num]) # the higher the more similar
            scores_subj.append(similarity_scores)
            pairs_subj.append(matched_pairs)
    scores_roi.append(scores_subj)
    pairs_roi.append(pairs_subj)
scores_roi = np.array(scores_roi)
pairs_roi = np.array(pairs_roi)

# visualization - strip plot
data = scores_roi.reshape((scores_roi.shape[0],-1))

# Generate meshgrid for region and sample indices
region_ids, sample_ids = np.meshgrid(
    np.arange(data.shape[0]),  # 8 regions
    np.arange(data.shape[1]),  # 120 samples
    indexing='ij'
)

# Convert meshgrid to DataFrame for plotting
df = pd.DataFrame({
    'ROIs': region_ids.flatten(),
    'Similarity': data.flatten()
})

# Assign ROI labels
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
df['ROIs'] = df['ROIs'].map(dict(enumerate(ROIs_label)))

# Set theme and create plot
sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(8, 5))
sns.despine(bottom=True, left=True, ax=ax)

# Stripplot (remove legend argument, not supported)
sns.stripplot(
    data=df, x='Similarity', y='ROIs',
    jitter=True, dodge=True, alpha=.25, zorder=1, ax=ax
)

# Pointplot (remove scale, use markersize instead; legend argument not supported)
sns.pointplot(
    data=df, x="Similarity", y="ROIs",
    dodge=True,
    join=False,
    markers="d",
    errorbar=None,  # replaces ci=None in recent seaborn
    linestyles="",  # no lines
    ax=ax
)
ax.set_ylabel('Regions of Interest', fontdict={'size':15})
ax.set_xlabel('Cosine Similarity', fontdict={'size':15})
ax.set_title(Paradigm+'-'+freqb, fontdict={'size':15})
plt.tight_layout()
plt.show()

# f.savefig('G:\CUHK_Intern\RESULTS/additionalResults/' + 'neuralModePrePostSimilarity_' + Paradigm + '_' + freqb + '.eps', format='eps', dpi=1000)
plt.close(f)


# also visualize the neural mode shift (pairs)
pairs_data = pairs_roi.reshape((pairs_roi.shape[0],pairs_roi.shape[1]*pairs_roi.shape[2],pairs_roi.shape[3]))
facecolors = ['#82B0D2','#FA7F6F','#6AB5AE','#B39CD8','#FFD56B','#F5CBA7','#6C7A89','#8FB68E','#F2A2B1','#A3C6D8']
edgecolors = ['#4F88B8','#E35A4F','#458D84','#9269B0','#D9B032','#D4A079','#4A5562','#668F6B','#C47483','#7CA4B8']
ROIs_label = ['PreCG.L','PreCG.R','SMA.L','SMA.R','SPG.L','SPG.R','IPL.L','IPL.R']
f1, ax1 = plt.subplots(figsize=(8, 5))
for roi_i in range(len(ROIs)):
    jitter1 = np.random.uniform(-0.5, 0.5, pairs_data.shape[1])
    jitter2 = np.random.uniform(-0.5, 0.5, pairs_data.shape[1])
    ax1.scatter(pairs_data[roi_i,:,0]+jitter1,pairs_data[roi_i,:,1]+jitter2,c=facecolors[roi_i],edgecolors=edgecolors[roi_i],s=20,label=ROIs_label[roi_i],alpha=0.6)

ax1.plot([0,20],[0,20],'k--',alpha=0.5,linewidth=1.5)
ax1.plot([5.5,20.5],[0,15],'k--',alpha=0.5,linewidth=1.5)
ax1.plot([0,15],[5.5,20.5],'k--',alpha=0.5,linewidth=1.5)
ax1.grid(True,linestyle='--',alpha=0.3)
ax1.set_xlabel('Neural Modes Order Before Training')
ax1.set_ylabel('Neural Modes Order After Training')
ax1.set_title('Neural Modes Shift '+ Paradigm + '-' + freqb)
plt.legend(frameon=False, bbox_to_anchor=(1.01,1), fontsize=10)
plt.tight_layout()
plt.show()
save_path = 'G:\CUHK_Intern\RESULTS/additionalResults/'
# f1.savefig(save_path + 'neuralModesShift_'+ Paradigm + '_' + freqb + '.eps', dpi=1000, format='eps')
plt.close(f1)


# obtain FMA scores
fma_scores_all = []
for i in range(2):
    fma_scores = []
    for subj in subj_list:
        fma_scores.append(subj_fma[i + 1][0][subj_fma[0][0] == subj].values)
    fma_scores_all.append(fma_scores)

fma_scores_all = np.array(fma_scores_all)
fma_diff = np.squeeze(fma_scores_all[1] - fma_scores_all[0])
fma_diff = np.concatenate((fma_diff[:4],fma_diff[5:]),axis=0)

var_stage = np.array(var_stage)
var_ = np.concatenate((var_stage[:,:,:4,:],var_stage[:,:,5:,:]),axis=2)

var_diff = np.zeros((len(ROIs),var_.shape[2],var_.shape[-1]))
for roi_i in range(len(ROIs)):
    for subj_i in range(var_.shape[2]):
        for comp_i in range(var_.shape[-1]):
            var_diff[roi_i,subj_i,comp_i] = var_[0,roi_i,subj_i,pairs_roi[roi_i,subj_i,comp_i,0]] - var_[1,roi_i,subj_i,pairs_roi[roi_i,subj_i,comp_i,1]]

# correlation
corr = []
p_value = []
power_neuralModeFMA = []
for roi_num in range(len(ROIs)):
    temp = np.mean(np.abs(var_diff[roi_num,:,:]),-1)

    res = spearmanr(temp, fma_diff)
    pow_, _ = stat_power(res.correlation+0.000000000000001, sample_size=fma_diff.shape[0])
    corr.append(res.correlation)
    p_value.append(res.pvalue)
    power_neuralModeFMA.append(pow_)

with open('G:\CUHK_Intern\RESULTS/additionalResults\powerAndEffectSize/neuralmode/neuralModeDiffFMA_power_'+Paradigm+'_'+freqb+'.txt', 'w') as f:
    for item in power_neuralModeFMA:
        f.write(f"{item}\n")

with open('G:\CUHK_Intern\RESULTS/additionalResults/neuralModeChangePrePostCorrFMA_' + Paradigm + '_' + freqb + '_corr.txt', 'w') as f:
    for item in corr:
        f.write(f"{item}\n")

with open('G:\CUHK_Intern\RESULTS/additionalResults/neuralModeChangePrePostCorrFMA_' + Paradigm + '_' + freqb + '_pvalue.txt', 'w') as f:
    for item in p_value:
        f.write(f"{item}\n")
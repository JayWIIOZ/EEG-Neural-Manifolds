import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *


# better to adjust the length of data, 20 may be enough!

# trial_num = 100
ROIs = [1]
Paradigm = 'AO1'
freqb = 'alpha'
train_stage = ['pre','post']
threshold = 1 # 0 - 1

load_path = 'chronic_stroke/pca_data/'
# subj_list = os.listdir(load_path)
subj_list = ['kmt','ock']
save_path = 'EEG-Neural-Manifolds/analysis/chronic_stroke/results/trajectory/'
os.makedirs(save_path, exist_ok=True)

GCCA_score = []
for roi in ROIs:
    GCCA_score_stage = []
    # data_list_stage = []
    for trainStage in train_stage:
        data_subj_list = []
        for subj in subj_list:
            data_path = os.path.join(load_path, trainStage, Paradigm, subj, str(roi), f'{subj}_pca_trial_{freqb}.npy')
            data_pca = np.load(data_path)
            time_len = data_pca.shape[1]
            rank = min(np.linalg.matrix_rank(data_pca))
            data_subj_list.append(np.reshape(data_pca[:,:,:rank], (-1, rank)))

        time_min = min([data.shape[0] for data in data_subj_list])
        rank_min = min([data.shape[1] for data in data_subj_list])
        data_subj_list_ = [data[:time_min, :rank_min] for data in data_subj_list]

        # aligning across each 2 subjects
        subj_pair = divide_pair(data_subj_list_)
        # canonical correlation
        CCA_score = []
        data_aligned = []
        for temp in subj_pair:
            A1, B1, r1, *_ = canoncorr(data_subj_list_[temp[0]], data_subj_list_[temp[1]], fullReturn=True)
            # *_, r1, U1, V1 = canoncorr(data_subj_list_[temp[0]], data_subj_list_[temp[1]], fullReturn=True)
            CCA_score.append(r1)
            # data_aligned.append([np.reshape(U1, (-1, time_len, U1.shape[-1])), np.reshape(V1, (-1, time_len, U1.shape[-1]))])
            U1, s1, Vh1 = svd(A1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U2, s2, Vh2 = svd(B1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            data_aligned.append([np.reshape(data_subj_list_[temp[0]], (-1, time_len, r1.shape[-1])) @ U1 @ Vh1,
                                 np.reshape(data_subj_list_[temp[1]], (-1, time_len, r1.shape[-1])) @ U2 @ Vh2])
        # data_list_stage.append(data_aligned)

        colors = get_colors(8, colormap='Paired')
        for pair_draw in range(len(subj_list)):
            data_aligned_avg = []
            for temp in data_aligned[pair_draw]:
                data_aligned_avg.append(np.mean(temp,axis=0))
            fig1 = plt.figure()
            ax1 = plt.axes(projection='3d', fc='None')
            for i in range(len(data_aligned_avg)):
                ax1.plot(data_aligned_avg[i][:, 0], data_aligned_avg[i][:, 1], data_aligned_avg[i][:, 2],color=colors[-2*(i+1)])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])
            ax1.set_xlabel('CC1',fontsize=15)
            ax1.set_ylabel('CC2',fontsize=15)
            ax1.set_zlabel('CC3',fontsize=15)
            ax1.set_title(subj_list[subj_pair[pair_draw][0]] + ' x ' + subj_list[subj_pair[pair_draw][1]],fontsize=15)
            fig1.savefig(save_path + subj_list[subj_pair[pair_draw][0]] + '-' + subj_list[subj_pair[pair_draw][1]] +
                         '_Region_' + str(roi) + '_' + freqb + '_' + trainStage + '_aligned.png', format='png', dpi=1000)
            fig1.show()
            
    # data_aligned_avg = [np.mean(temp, axis=0) for temp in data_aligned[4]]
    # for i in range(len(subj_pair[4])):
    #     fig1 = plt.figure(i)
    #     ax1 = plt.axes(projection='3d', fc='None')
    #     ax1.plot(data_aligned_avg[i][:, 0], data_aligned_avg[i][:, 1], data_aligned_avg[i][:, 2])
    #     ax1.set_xticks([])
    #     ax1.set_yticks([])
    #     ax1.set_zticks([])
    #     ax1.set_xlabel('CC1')
    #     ax1.set_ylabel('CC2')
    #     ax1.set_zlabel('CC3')
    #     ax1.set_title(subj_list[subj_pair[4][i]] + '-Region ' + str(roi) + '-' + freqb + '-' + trainStage)
    #     fig1.show()
    #     fig1.savefig(save_path + subj_list[subj_pair[4][i]] + '_Region_' + str(roi) + '_' + freqb + '_' + trainStage
    #                  + '_aligned.eps', format='eps', dpi=1000)


    # gcca = GCCA(latent_dimensions=20)
    # gcca.fit(data_subj_list_)
    # GCCA_score.append(gcca.average_pairwise_correlations(data_subj_list_))
    #
    # # align across trials
    # data_aligned = [data_ @ gcca.weights_[i] for i, data_ in enumerate(data_subj_list_)]
    # data_aligned_ = [np.reshape(data__, (-1, time_len, data__.shape[-1])) for data__ in data_aligned]
    # data_aligned = []
    # for i in range(len(data_subj_list_)):
    #     U, s, Vh = svd(gcca.weights_[i], full_matrices=False, compute_uv=True, overwrite_a=False,
    #                    check_finite=False)
    #     data_aligned.append(data_subj_list_[i] @ U @ Vh)
    # data_aligned_ = [np.reshape(data__, (-1, time_len, data__.shape[-1])) for data__ in data_aligned]



        # for ii in range(3):
        #     fig2, ax2 = plt.subplots(ncols=1)
        #     ax2.plot(data_aligned_avg[0][:,ii])
        #     ax2.set_xticks([])
        #     ax2.set_yticks([])
        #     fig2.show()
        #     fig2.savefig(save_path + subj_list[subj_pair[5][0]] + '_Region_' + str(roi) + '_' + freqb + '_' + trainStage
        #      + '_' + str(ii) +'.eps', format='eps', dpi=1000)

# GCCA_score_stage.append(GCCA_score)
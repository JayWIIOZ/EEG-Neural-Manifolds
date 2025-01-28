import numpy as np
import matplotlib.pyplot as plt
from utils import *

# load data
ROIs = [1]
train_stage = ['pre','post']
Paradigm = 'AO1'
freqb = 'theta'

load_path = 'F:/CUHK_intern/RESULTS/Multimodality/'
subj_list = ['kmt','wws','nsk','nwc','wsc','ock','wwf']
save_path = 'F:\CUHK_Intern\RESULTS/figure\Multimodality/trajectory/'

for roi in ROIs:
    for trainStage in train_stage:
        data_subj_list = []
        for subj in subj_list:
            data_path = load_path + trainStage + '/' + Paradigm + '/' + subj + '/trial/' + str(roi) + '/'
            data_pca = np.load(
                data_path + subj + '_' + Paradigm + '_' + trainStage + '_pca_trial_' + freqb + '.npy')
            data_subj_list.append(data_pca)

        subj_pair = divide_pair(data_subj_list)

        colors = get_colors(8, colormap='Set2')
        for pair_draw in range(len(subj_list)-1):
            data_avg = []
            for temp in subj_pair[pair_draw]:
                data_avg.append(np.mean(data_subj_list[temp],axis=0))
            fig1 = plt.figure()
            ax1 = plt.axes(projection='3d', fc='None')
            for i in range(len(data_avg)):
                ax1.plot(data_avg[i][:, 0], data_avg[i][:, 1], data_avg[i][:, 2], color=colors[2*i])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])
            ax1.set_xlabel('CC1', fontsize=15)
            ax1.set_ylabel('CC2', fontsize=15)
            ax1.set_zlabel('CC3', fontsize=15)
            ax1.set_title(subj_list[subj_pair[pair_draw][0]] + ' x ' + subj_list[subj_pair[pair_draw][1]], fontsize=15)
            fig1.show()
            fig1.savefig(save_path + subj_list[subj_pair[pair_draw][0]] + '-' + subj_list[subj_pair[pair_draw][1]] +
                         '_Region_' + str(roi) + '_' + freqb + '_' + trainStage + '_unaligned.eps', format='eps',
                         dpi=1000)

        # data_pca_pre_avg = np.mean(data_pca_pre, axis=0)
        # fig1 = plt.figure(2*i)
        # ax1 = plt.axes(projection='3d', fc='None')
        # ax1.plot(data_pca_pre_avg[ :, 0], data_pca_pre_avg[:, 1], data_pca_pre_avg[:, 2])
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # ax1.set_zticks([])
        # ax1.set_xlabel('PC1')
        # ax1.set_ylabel('PC2')
        # ax1.set_zlabel('PC3')
        # ax1.set_title(subj+'-Region '+str(roi)+'-'+freqb+'-Pre')
        # fig1.show()
        # fig1.savefig(save_path + subj +'_Region_' + str(roi) + '_' + freqb + '_pre.eps', format='eps', dpi=1000)
        #
        # data_pca_post_avg = np.mean(data_pca_post, axis=0)
        # fig2 = plt.figure(2 * i + 1)
        # ax2 = plt.axes(projection='3d', fc='None')
        # ax2.plot(data_pca_post_avg[:, 0], data_pca_post_avg[:, 1], data_pca_post_avg[:, 2])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # ax2.set_zticks([])
        # ax2.set_xlabel('PC1')
        # ax2.set_ylabel('PC2')
        # ax2.set_zlabel('PC3')
        # ax2.set_title(subj+'-Region '+str(roi)+'-'+freqb+'-Post')
        # fig2.show()
        # fig2.savefig(save_path + subj +'_Region_' + str(roi) + '_' + freqb + '_post.eps', format='eps', dpi=1000)
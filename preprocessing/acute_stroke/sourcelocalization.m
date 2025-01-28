%% batch source localization of preprocessed EEG data
clear;clc;
addpath(genpath('/Users/cizer/Downloads/taoliu/fieldtrip-20250114'))
% load brainnetome atlas
atlas = ft_read_atlas('/Users/cizer/Downloads/taoliu/fieldtrip-20250114/template/atlas/aal/ROI_MNI_V4.nii');
atlas = ft_convert_units(atlas, 'mm');
freqBand.freq = [[1 3];[4 8];[8 13];[13 30];[30 40]];
freqBand.freqLabel = {'detla','theta','alpha','beta','gamma'};

paradim = 'AO';
ROIs = [1 2 19 20 59 60 61 62];

data_dir = '/Users/cizer/Downloads/taoliu/newcode/EEG-Neural-Manifolds/dataset/acute_stroke_dataset/edffile';

subj_list = dir(data_dir);
%% 
for subj_num = 1:length(subj_list)
    subj_name = subj_list(subj_num+3).name;

    % load headmodel
    load('/Users/cizer/Downloads/taoliu/fieldtrip-20250114/template/headmodel//standard_bem.mat');

    % load electrode information
    load('/Users/cizer/Downloads/taoliu/newcode/EEG-Neural-Manifolds/tool/elec_realigned_1.mat');

    % load preprocessed data
    EEG = pop_loadset([data_dir,'/',subj_name,'/eeg/setfiles/',subj_name,'_preprocessed.set']);
    EEG = eeg_checkset( EEG );
    data = eeglab2fieldtrip(EEG, 'preprocessing');

    % latency selection
    cfg = [];
    cfg.latency = [0 4];
    data = ft_selectdata(cfg, data);

     % forward solution
    cfg = [];
    cfg.elec =  elec_realigned;     % sensor positions
    % cfg.elec =  freqAll.elec;
    cfg.headmodel = vol;        % volume conduction model
    cfg.reducerank      = 3;
    cfg.channel         = 'all';
    cfg.grid.resolution = 0.1;   % use a 3-D grid with a 10 mm resolution
    cfg.grid.unit       = 'dm';
    % cfg.grid    = sourcemodel;
    % leadfield = ft_prepare_leadfield(cfg);
    leadfield = ft_prepare_leadfield(cfg,data);

    % for trial_num = 1:length(data.trial)
    for trial_num = 1:length(data.trial) % this is a hard code, better to determine based on actual data    
        % compute covariance matrix
        cfg = [];
        cfg.covariance ='yes';  
        cfg.vartrllength = 2; % dont change
        cfg.trials = trial_num;
        data_timelock= ft_timelockanalysis(cfg,data);
        
        % backward solution
        cfg                   = [];
        %cfg.frequency         = 30;
        cfg.method            = 'eloreta';
        cfg.grid              = leadfield;
        cfg.headmodel         = vol;
        % cfg.keeptrials        = 'yes';
        %     cfg.bootstrap         = 'yes';
        % cfg.rawtrial          = 'yes';
        %     cfg.mne.lambda    = '10%';
        cfg.eloreta.projectnoise  = 'yes';
        cfg.eloreta.keepfilter    = 'yes';
        % cfg.eloreta.keepcsd       = 'yes'; % this line does not make sense, eloreta does not give csd
        cfg.eloreta.fixedori      = 'yes';
        sourceAll = ft_sourceanalysis(cfg, data_timelock);
        
        % 3D to 1D
        cfg = [];
        cfg.projectmom = 'yes';
        source = ft_sourcedescriptives(cfg, sourceAll);
        
        % create voxel * samples matrix, with empty voxel values NaN
        source.avg.momint = nan(size(source.pos,1), length(source.time));
        for j=1:length(source.inside)
          indx = source.inside(j);
          if indx == 1
              source.avg.momint(j,:) = source.avg.mom{j};
              % the grid locations outside the brain keep their NaN
          end
        end
    
        clear data_timelock sourceAll
    
        % source activity to voxel activity
        cfg              = [];
        cfg.parameter    = 'momint';
        cfg.interpmethod = 'nearest';
        % cfg.sphereradius = 0.5;
        source_int  = ft_sourceinterpolate(cfg,source,atlas); % interpolate based on atlas

        save(['D:\code\dataset\acute_storke\RESULTS\Multimodality\',stage,'\',paradim,'\',subj_name,'\','trial\',subj_name,'_',paradim, ...
              '_source_',num2str(trial_num),'.mat'],'source_int'); % dont need to save source_int

        clear source

        % finding voxels belong to motor cortex
        cfg = [];
        tf_source_seg = sourcesegmentation(cfg, source_int, atlas);

        for roi_i = 1:length(ROIs)
            roi = ROIs(roi_i);
            mkdir(['D:\code\dataset\acute_storke\RESULTS\Multimodality\',subj_name,'\','trial\',num2str(roi)]);
            momint_1 = source_int.momint(find(tf_source_seg==roi),:); % save source_int.momint and tf_source_seg instead
            
            if mod(roi,2) == 0
                save(['E:\taoliu\RESULTS\Multimodality\',subj_name,'\','trial\',num2str(ROIs(roi_i)),'\',subj_name,'_voxel_',num2str(trial_num),'_l.mat'],'momint_1');
            else
                save(['E:\taoliu\RESULTS\Multimodality\',subj_name,'\','trial\',num2str(ROIs(roi_i)),'\',subj_name,'_voxel_',num2str(trial_num),'_r.mat'],'momint_1');
            end
        end
        clear source_int
    end
    clear data leadfield vol
end
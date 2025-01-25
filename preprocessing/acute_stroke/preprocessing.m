clear; close all; clc;
addpath(genpath('D:\code\toolbox\fieldtrip-20240515')); 
addpath(genpath('eeglab_path'));
mkdir('D:\code\Integrated_Output\acute_stroke');

data_dir = 'D:\code\dataset\acute_stroke\edffile';
subj_list = dir(data_dir);

ROIs = [1 2 19 20 59 60 61 62];
paradim = 'AO'; 
%% preprocessing
for i = 1:length(subj_list)
    subj_name = subj_list(i).name;
    filepath = [data_dir,'\',subj_name,'\eeg\'];
    File = dir(fullfile(filepath,'*.edf'));
    filenames = {File.name};
    setname = filenames{1};
    EEG.setname = setname(1:6);
    EEG = eeg_checkset( EEG );
    EEG = pop_biosig([filepath,setname]);
    EEG = pop_chanedit(EEG, 'load',{'D:\code\dataset\acute_stroke\task-motor-imagery_electrodes.tsv','filetype','autodetect'});
    mkdir([filepath,'\','setfiles']);
    set_dir = [filepath,'\','setfiles'];
    EEG.setname = setname(1:6);
    EEG = pop_saveset( EEG, 'filename',[EEG.setname,'.set'],'filepath',set_dir);

    %% define events
    for ii = 1:length([EEG.event])
        if string(EEG.event(ii).type) == '1'
            EEG.event(ii).type = 'IS';
        elseif string(EEG.event(ii).type) == '2'
            EEG.event(ii).type = 'MI';
        elseif string(EEG.event(ii).type) == '3'
            EEG.event(ii).type = 'B';
        else
            EEG.event(ii).type = 'else';
        end
    end
    %% reject channels
    originalEEG = EEG;
    EEG = clean_rawdata(EEG, -1, [-1], 0.8, -1, -1, -1);
    %% Interpolate all the removed channels
    EEG = pop_interp(EEG, originalEEG.chanlocs, 'spherical');
    EEG = pop_reref( EEG, []);

    EEG = pop_select( EEG, 'rmchannel',{'HEOL','VEOR'});
    EEG = pop_reref( EEG, []);
    EEG = clean_rawdata(EEG, -1, [-1], -1, -1, 10, -1);
    EEG = pop_saveset( EEG, 'filename',[EEG.setname,'_artifactRemove.set'],'filepath',set_dir);

    %% amica
    load_path = set_dir;
    out_path = [set_dir,'\'];
    cd(load_path);
    test_amica([EEG.setname,'_artifactRemove.set'],load_path,out_path);

    close all

end

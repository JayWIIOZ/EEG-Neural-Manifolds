clear
close all
clc

data_dir = '/Users/cizer/Downloads/taoliu/newcode/EEG-Neural-Manifolds/dataset/acute_stroke_dataset/edffile';

%% load data
subj_list = dir(data_dir);

        
        for i = 1:length(subj_list)
            subj_name = subj_list(i+3).name;
   
            %% load data and remove artifacts
            filepath = [data_dir,'/',subj_name,'/eeg/setfiles/'];
            % File = dir(fullfile(filepath,['*artifactsRemove.set']));
            File = dir(fullfile(filepath,['*artifactRemove.set']));
            filenames = {File.name}';
            setname = filenames{1};
            savename = [subj_name,'_preprocessed.set'];
            EEG = pop_loadset('filename',setname,'filepath',filepath);
            EEG = eeg_checkset(EEG);
            EEG = pop_resample( EEG, 100);
            EEG = pop_epoch( EEG, {  'MI'  }, [0  6], 'newname', ...
                [subj_name,'_artifactsRemove.set resampled epochs'], 'epochinfo', 'yes');
            EEG = pop_rmbase( EEG, [4000 5990] ,[]);

            EEG = pop_iclabel(EEG, 'default');
            EEG = pop_icflag(EEG, [NaN NaN;0.5 1.1;0.7 1;NaN NaN;NaN NaN;0.8 1;NaN NaN]);
            EEG = pop_subcomp( EEG, [], 0);
            EEG = eeg_checkset( EEG );

            EEG = pop_iclabel(EEG, 'default');
            EEG = pop_icflag(EEG, [NaN NaN;0.4 1;0.7 1;NaN NaN;NaN NaN;0.7 1;NaN NaN]);
            EEG = pop_subcomp( EEG, [], 0);
            EEG = eeg_checkset( EEG );
            
            EEG = pop_select(EEG,'nochannel',{'M1','M2','CB1','CB2','I1','I2','TP9','TP10'});
            EEG = pop_reref( EEG, []);

            EEG = pop_saveset( EEG, 'filename',savename,'filepath',filepath);
                    
            close all
            % test_amica('kmt_artifactsRemove.set','D:\CUHK_Intern\Angels_EEG\Preprocessed\pre\kmt\','D:\CUHK_Intern\Angels_EEG\Preprocessed\pre\kmt\')
        end
    end 
end
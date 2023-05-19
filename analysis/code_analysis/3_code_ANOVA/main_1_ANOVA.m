run('../setup_analysis.m')
%% anova
W_lp.loop('function_ANOVA', {'dataset_withV'}, 'anova.mat');
%% keep only sig units
W_lp.loop('function_cleanspikes', {'dataset_withV', 'anova'}, {'data_cleaned', 'spikes_cleaning_info'}, 'pos_data', [1, -1], 0, 0.2, 0.05);



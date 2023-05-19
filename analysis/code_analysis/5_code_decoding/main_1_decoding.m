run('../setup_analysis.m')
%% decoding by bin
W_lp.overwrite_on;
W_lp.parloop('function_decoding', {'pca', 'data_cleaned'}, 'decoding', 20);
%% decoding
W_lp.parloop('function_decoding_overall', {'pca', 'data_cleaned'}, 'decoding_overall', 20);


run('../setup_analysis.m')
%% PCA
W_lp.overwrite_on;
W_lp.parloop('function_PCA_projection', {'data_cleaned'}, 'pca');
W_lp.overwrite_off;

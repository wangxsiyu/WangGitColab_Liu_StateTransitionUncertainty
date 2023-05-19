run('../setup_analysis.m')
W_lp.overwrite_on
W_lp.parloop('function_1D_projection', {'pca', 'data_cleaned'}, 'x1D');

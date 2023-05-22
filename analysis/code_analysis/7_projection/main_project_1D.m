run('../setup_analysis.m')
W_lp.overwrite_off
W_lp.loop('function_1D_projection', {'pca', 'data_cleaned'}, 'x1D');

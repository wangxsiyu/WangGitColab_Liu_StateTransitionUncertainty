run('../setup_analysis.m')
%%
W_lp.overwrite_on;
W_lp.parloop('function_energy_landscape', {'data_cleaned', 'x1D'}, ...
    'EL')
W_lp.overwrite_off;
%%
run('../setup_plot.m')
%%
data = W_lp.load({'decoding', 'data_cleaned'}, [], 'cell');
%%
wv_decode = W.cellfun(@(x)x, data(:,1), false);

%%
vars = {'choice', 'highstate','trans', 'reward', 'observedplanet'};
for vari = 1:length(vars)
    varname = vars{vari};
    plt.figure(1,1, 'is_title', 1);
    ac_svm = W.cellfun(@(x)x.(['ac_' varname]), wv_decode, false);
    [av, se] = W.cell_avse(ac_svm);
    plt.setfig_ax('xlabel', 'time (ms)', 'ylabel', 'decoding accuracy', ...
        'title', varname);
    plt.plot(1:300, av, se, 'shade', 'color', plt.custom_vars.color_cond);
    plt.dashX(0.5);
    plt.update(['decode_' varname], ' ')
end
%%

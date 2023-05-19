%%
data = W_lp.load({'anova', 'data_cleaned'}, [], 'cell');
%%
wv_anova = W.cellfun(@(x)x, data(:,1), false);
%%
run('../setup_plot.m')
%%
plt.set_custom_variables('color_anova', {'RSred','AZcactus','AZsand','AZsky'})
plt.figure(1,1);
plt.setfig_ax('ylim', [0 1], 'ytick', 0:.5:1, ...
    'xlim',[.5 300.5]);
plt.FIG_ANOVA(wv_anova, [], 1:300);
plt.update;
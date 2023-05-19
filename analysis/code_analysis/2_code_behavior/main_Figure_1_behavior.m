run('../setup_plot.m')
%%
gp = W.analysis_1group(sub);
%%
xlab = plt.custom_vars.xlabel_cond;
cond = plt.custom_vars.name_cond;
cols = plt.custom_vars.color_cond;
%%
plt.figure;
plt.setfig(1,'xlabel', xlab, 'ylabel', 'p(high state)', ...
    'ylim', [0 1], 'legloc', 'SW');
plt.ax(1);
vn = 'avCORRECT_byCOND';
nT = size(gp.(['GPav_', vn]){1},2);
plt.plot(1:nT, gp.(['GPav_', vn]){1}, gp.(['GPste_', vn]){1}, 'shade', 'color', cols);
plt.setfig_ax('legend', arrayfun(@(x)sprintf("pT = %.0f, pA = %.0f", gp.GPav_gpT{1}(x), gp.GPav_gpA{1}(x)),1:length(gp.GPav_gpA{1})));
plt.update('pcorrect', ' ');
%%
plt.figure;
tav = [gp.GPav_Q1_LL; gp.GPav_MB_LL];
tse = [gp.GPste_Q1_LL; gp.GPste_MB_LL];
plt.plot(1:size(tav,2), tav(2:-1:1,:), tse(2:-1:1,:), 'line', 'color', plt.custom_vars.col_MBMF);
plt.setfig_ax('legend', plt.custom_vars.leg_MBMF, 'legloc', 'SW', ...
    'xlabel', cond, 'ylabel', 'log likelihood', 'xlim',[0.5, 5.5], ...
    'xtick', 1:5, 'xticklabel', W.iif(cond == 'ambiguity', gp.GPav_gpA{1}', gp.GPav_gpT{1}'));
plt.update('logll','  ');
%%
plt.figure;
plt.plot([], gp.GPav_mix_ratioMB, gp.GPste_mix_ratioMB, 'line', 'color','black');
plt.setfig_ax('legend', plt.custom_vars.leg_MBMF, 'legloc', 'SW', ...
    'xlabel', cond, 'ylabel', 'weight (MB)', 'xlim',[0.5, 5.5], ...
    'xtick', 1:5, 'xticklabel', W.iif(cond == 'ambiguity', gp.GPav_gpA{1}', gp.GPav_gpT{1}'));
plt.update('MLEratio','  ');

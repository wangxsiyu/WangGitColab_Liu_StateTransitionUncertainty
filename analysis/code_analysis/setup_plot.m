run('./setup_analysis.m')
sub = W_lp.load('sub');
sub = cellfun(@(x)x, sub);
sub = struct2table(sub);
% function plt = setup_plot(figdir)
figdir = W.get_fullpath(W.mkdir('./Figures'));
plt = W_plt('savedir', fullfile(figdir,versionname), ...
    'issave', true, 'extension', {'jpg'}, ...
    'isshow', true);
cols = plt.translatecolors({'AZred', 'AZsand'});
col_T = W.arrayfun(@(x)plt.interpolatecolors(cols, [0, 1], x), [0 0.25 0.5 0.75 1]);

cols = plt.translatecolors({col_T{4}, 'AZblue'});
col_A = W.arrayfun(@(x)plt.interpolatecolors(cols, [0, 1], x), [0 0.25 0.5 0.75 1]);
plt.set_custom_variables('color_ambiguity', col_A, ...
    'color_uncertainty', col_T, 'leg_MBMF', {'MB', 'MF'}, ...
    'col_MBMF', {'AZred', 'AZblue'});
% end
namecond = W.file_getprefix(versionname);
xlabcond = W.file_getsuffix(versionname);
switch xlabcond
    case 'trialnumber'
        xlabcond = 'trial number';
    case 'reversal'
        xlabcond = 'trial since reversal';
end
plt.set_custom_variables('color_cond', ...
    plt.custom_vars.(sprintf('color_%s', namecond)), ...
    'name_cond', namecond, ...
    'xlabel_cond', xlabcond);

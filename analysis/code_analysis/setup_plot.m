run('./setup_analysis.m')
sub = W_lp.load('sub');
sub = cellfun(@(x)x, sub);
sub = struct2table(sub);
% function plt = setup_plot(figdir)
figdir = W.get_fullpath(W.mkdir('./Figures'));
plt = W_plt('savedir', fullfile(figdir,versionname), ...
    'issave', true, 'extension', {'jpg'}, ...
    'isshow', true);
% cols = plt.translatecolors({'RSyellow', 'AZred'});
% col_T = W.arrayfun(@(x)plt.interpolatecolors(cols, [0, 1], x), [0 0.25 0.5 0.75 1]);
col_T = {[253,218,13]/255,[255,160,122]/255,[240,128,128]/255,[220,20,60]/255,[255,0,0]/255};
% cols = [col_T(3), plt.translatecolors({'RSgreen', 'AZblue'})];
% col_A = W.arrayfun(@(x)plt.interpolatecolors(cols, [0, 0.5, 1], x), [0 0.25 0.5 0.75 1]);
col_A = {[0,0,139]/255,[0,0,1],[65,105,225]/255,[0,191,255]/255,[176,224,230]/255};
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
switch namecond
    case 'ambiguity'
        namecond1 = 'p(ambiguity)';
    case 'uncertainty'
        namecond1 = 'p(common)';
end
plt.set_custom_variables('color_cond', ...
    plt.custom_vars.(sprintf('color_%s', namecond)), ...
    'name_cond', namecond, ...
    'str_cond', namecond1, ...
    'xlabel_cond', xlabcond);

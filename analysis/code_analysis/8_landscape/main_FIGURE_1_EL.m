run('../setup_plot.m');
%%
EL = W_lp.load({'EL'},[],'cell');
time_EL = EL{1}.time_EL;
%%
x1D = W_lp.load({'x1D'},[],'cell');
time_at = x1D{1}.time_at;
x1D = W.cellfun(@(x)x.x1D, x1D);
%%
games = W_lp.load('data_cleaned');
games = W.cellfun(@(x)x.games, games);
%%
%%
plt.figure(2,3,'is_title', 1, 'gapW_custom', [0 1 1 1] * 50, 'matrix_hole', [1 1 1; 1 1 1]);
plt.reload_paramdatabase();
plt.param_scale(1,[],1,1.3);
cond = plt.custom_vars.name_cond;
tlt =  arrayfun(@(x)sprintf("pT = %.0f, pA = %.0f", gp.GPav_gpT{1}(x), gp.GPav_gpA{1}(x)),1:length(gp.GPav_gpA{1}));
plt.setfig(1:5,'title', W.str2cell(tlt));
FIG_Energy_landscape_over_time(plt, x1D, time_at,EL, games, sub.animal, [-10, 5]);

FIG_Energy_landscape_by_cue_new(plt, EL, sub, 20)

plt.update('development', 'FGHIJK');


% 
% %%
% timeslice = 'median';
% time_EL = EL{1}.time_EL;
% x_EL = EL{1}.x_EL;
% %% compile curves by cue
% EL_cue = cell(1, length(EL));
% for si = 1:length(EL)
%     if W.is_stringorchar(timeslice)
%         ttmslice = sub.midRT_REJECT(si) * 1000;
%     else
%         ttmslice = timeslice;
%     end
%     idxt = dsearchn(time_EL', ttmslice);
%     tEL = W.cellfun(@(t)t(idxt,:), EL{si}.grad_cue, false);
%     EL_cue{si} = vertcat(tEL{:});
% end
% %% average between monkeys
% mks = unique(sub.animal);
% avEL = {};
% seEL = {};
% pa = [];
% for i = 1:length(mks)
%     [avEL{i}, seEL{i}] = W.cell_avse(EL_cue(sub.animal == mks(i)));
%     % compute pa
%     pa(i,:) = mean(sub.avCHOICE_byCONDITION(sub.animal == mks(i),:));
% end
% %% development
% plt.figure(1,2,'is_title', 1, 'gapW_custom', [0 1 1] * 100);
% plt.setfig('title', W.str2cell(W.file_prefix(mks, 'Monkey',' ')));
% cols = plt.translatecolors({plt.custom_vars.color_rejectaccept{1},'yellow',plt.custom_vars.color_rejectaccept{2}});
% for i = 1:2
%     [~,od] = sort(pa(i,:));
%     condcolors = W.arrayfun(@(x)plt.interpolatecolors(cols, [0,.5,1], x), pa(i,:));
%     leg = arrayfun(@(x)sprintf("p = %.2f", x), pa(i,od));
%     plt.setfig_ax('xlabel', 'position', 'ylabel', 'V', ...
%         'legloc','eastoutside',...
%         'legend', leg, 'xlim', [-3, 3]);
%     plt.plot(x_EL, avEL{i}(od,:), seEL{i}(od,:),'line', 'color', condcolors(od));
%     plt.new;
% end
% plt.update([],'  ');
% plt.save(sprintf('Figure4 - energy landscape by cue at %sms', W.string(timeslice)));

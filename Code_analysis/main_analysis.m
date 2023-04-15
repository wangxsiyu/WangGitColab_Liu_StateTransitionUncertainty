datadir = '../training/data';
files = W.ls(datadir, 'dir');
%%
d = {};
tab = table;
for i = 1:length(files)
    W.print('load folder %d', i)
    tdir = files{i};
    d{i} = W.load(tdir, 'csv');
    d{i} = W.cellfun(@(x)preprocess_2step(x), d{i}, false);
    d{i} = W.cellfun(@(x)W.tab_fill(x, 'subjectID', W.basenames(tdir)), d{i}, false);
    tab = W.tab_vertcat(tab, d{i}{:});
end
%%
save('./data_v1/data.mat', "tab", '-v7.3');
%%
idx = W.selectsubject(tab, {'subjectID','p_major','p_reward_high','p_ambiguity'});
save('./data_v1/idx.mat', "idx", '-v7.3');
%%
load('./data_v1/data.mat');
load('./data_v1/idx.mat');
%%
sub = W.analysis_sub(tab, idx, {'MLE_2step', 'behavior_2step'});
sub.agenttype = W.strs_selectbetween2patterns(sub.subjectID, [], '_', [], 1);
%%  
save('./data_v1/sub.mat', 'sub', '-v7.3');
%%
gp0 = W.analysis_group(sub, {'agenttype', 'p_reward_high', 'p_ambiguity', 'p_major'});
%%
plt = W_plt('savedir', '../Figures');
%%
plt.figure(1,3, 'is_title',1);
plt.setfig_all('xlabel', 'trialID', 'ylabel', 'p(reward)', 'ylim', [0 1], 'legloc', 'SE');
plt.setfig(1:3,'title', {'p(reward)','p(common)','p(ambiguity)'});
gp = gp0;
plt.ax(1);
tgp = gp(gp.group_analysis.p_ambiguity == "0" & gp.group_analysis.p_major == "1",:);
[~,tod] = sort(tgp.GPav_p_reward_high);
tgp = tgp(tod,:);
cols = {'AZred50','AZred60','AZred70','AZred80','AZred90','AZred'};
plt.plot(1:30, tgp.GPav_avREWARD_byTRIALID, tgp.GPste_avREWARD_byTRIALID, 'line', 'color', cols);
plt.setfig_ax('legend', arrayfun(@(x)sprintf("pR = %.2f", x),tgp.GPav_p_reward_high));

plt.ax(2);
tgp = gp(gp.group_analysis.p_ambiguity == "0" & gp.group_analysis.p_reward_high == "1",:);
[~,tod] = sort(tgp.GPav_p_major );
tgp = tgp(tod,:);
cols = {'AZblue50','AZblue60','AZblue70','AZblue80','AZblue90','AZblue'};
plt.plot(1:30, tgp.GPav_avREWARD_byTRIALID, tgp.GPste_avREWARD_byTRIALID, 'line', 'color', cols);
plt.setfig_ax('legend', arrayfun(@(x)sprintf("pT = %.2f", x),tgp.GPav_p_major));

plt.ax(3);
tgp = gp(gp.group_analysis.p_major == "1" & gp.group_analysis.p_reward_high == "1",:);
[~,tod] = sort(tgp.GPav_p_ambiguity );
tgp = tgp(tod,:);
cols = {'AZcactus','AZcactus90','AZcactus80','AZcactus70','AZcactus60','AZcactus50'};
plt.plot(1:30, tgp.GPav_avREWARD_byTRIALID, tgp.GPste_avREWARD_byTRIALID, 'line', 'color', cols);
plt.setfig_ax('legend', W.arrayfun(@(x)sprintf("pA = %.2f", x),tgp.GPav_p_ambiguity, false));

plt.update;

% %%
% plt.figure(2,3, 'is_title',1);
% plt.setfig_all('xlabel', 'trialID', 'ylabel', 'p(correct)');
% plt.setfig(1:3,'title', {'p(reward)','p(common)','p(ambiguity)'});
% 
% agenttype = {'Ambiguity', 'Transition'};
% for i = 1:2
%     gp = gp0(gp0.group_analysis.agenttype == agenttype{i},:);
%     plt.ax(i,1);
%     tgp = gp(gp.group_analysis.p_ambiguity == "0" & gp.group_analysis.p_major == "1",:);
%     [~,tod] = sort(tgp.GPav_p_reward_high);
%     tgp = tgp(tod,:);
%     cols = {'AZred50','AZred60','AZred70','AZred80','AZred90','AZred'};
%     plt.plot(1:30, tgp.GPav_avREWARD_byTRIALID, tgp.GPste_avREWARD_byTRIALID, 'line', 'color', cols);
%     plt.setfig_ax('legend', arrayfun(@(x)sprintf("pR = %.2f", x),tgp.GPav_p_reward_high));
% 
%     plt.ax(i,2);
%     tgp = gp(gp.group_analysis.p_ambiguity == "0" & gp.group_analysis.p_reward_high == "1",:);
%     [~,tod] = sort(tgp.GPav_p_major );
%     tgp = tgp(tod,:);
%     cols = {'AZblue50','AZblue60','AZblue70','AZblue80','AZblue90','AZblue'};
%     plt.plot(1:30, tgp.GPav_avREWARD_byTRIALID, tgp.GPste_avREWARD_byTRIALID, 'line', 'color', cols);
%     plt.setfig_ax('legend', arrayfun(@(x)sprintf("pT = %.2f", x),tgp.GPav_p_major));
% 
%     plt.ax(i,3);
%     tgp = gp(gp.group_analysis.p_major == "1" & gp.group_analysis.p_reward_high == "1",:);
%     [~,tod] = sort(tgp.GPav_p_ambiguity );
%     tgp = tgp(tod,:);
%     cols = {'AZcactus','AZcactus90','AZcactus80','AZcactus70','AZcactus60','AZcactus50'};
%     plt.plot(1:30, tgp.GPav_avREWARD_byTRIALID, tgp.GPste_avREWARD_byTRIALID, 'line', 'color', cols);
%     plt.setfig_ax('legend', arrayfun(@(x)sprintf("pA = %.2f", x),tgp.GPav_p_ambiguity));
% end
% plt.update;



files = W.dir('*.csv');
data = W.load(files.fullpath, 'csv');
%%
data = W.arrayfun(@(x)W.tab_fill(data{x}, 'subjectID', x), 1:length(data), false);
data = W.tab_vertcat(data{:});
%%
idxsub = W.selectsubject(data, {'subjectID','episodeID'});
%%
sub = W.analysis_sub(data, idxsub, 'be2');
%%
ss = W.analysis_group(sub, 'subjectID');
%%
gp = W.analysis_1group(ss);
%%
plt = W_plt;
plt.figure();
plt.plot(1:100,gp.GPav_GPav_avR, gp.GPste_GPav_avR, 'line');
plt.update;
%% 
plt.figure()
plt.plot(1:4, gp.GPav_GPav_bar, gp.GPste_GPav_bar, 'bar');
plt.update;
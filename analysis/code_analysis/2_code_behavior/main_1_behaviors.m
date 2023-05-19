run('../setup_analysis.m')
%%
[data] = W_lp.load('dataset.mat');
games = W.arrayfun(@(x)W.tab_fill(data{x}.games, 'filename', W_lp.folderIDs(x)), 1:length(data), false);
%% compile all games
games = W.tab_vertcat(games{:});
[idxsub] = W.selectsubject(games, 'filename');
%% calculate sub
sub = W.analysis_sub(games, idxsub, {'behavior_2step'}, 'parfor');
%% add basic information
sub.animal = W.file_getprefix(sub.filename);
sub.idx_animal = W.arrayfun(@(x) find(x == unique(sub.animal)), sub.animal);
%% save group behavior
savename = fullfile(datadir, 'behavior_sub');
W.save(savename, 'sub', sub);
%% distribute data to individual sessions
subs = W.arrayfun(@(x)table2struct(sub(x,:)), 1:size(sub,1), false);
W_lp.save(subs, 'sub', sub.filename);


%% update games
W_lp.overwrite_on;
W_lp.loop('calc_value', {'dataset', 'sub'}, 'dataset_withV');



% 
% %% add MB/MF Q value
% W_lp.loop('calc_value', {'sub','dataset'}, 'sub_V');
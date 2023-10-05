masterdir = '\\hpcdrive.nih.gov\data\model_twostep';
savedir = './simudata';
%%
folders = W.dir(fullfile(masterdir, 'Seed*'), 'dir');
folders = folders(contains(folders.filename, 'LSTM'), :);
%%
tab = table;
for i = 1:size(folders,1)
    subfd = W.ls(folders.fullpath(i));
    subfd = subfd(1);

    t = table;
    t.env = 'GA';
    savename = 'na';
    t.savename = fullfile(savedir, savename);
    t.n_episode = 100;
    t.modelname = "LSTM";
    t.modelfolder = subfd;
    t.is_record = 0;
    tab = W.tab_vertcat(tab, t);

end
%%
W.writetable(tab, 'record_plan.csv')
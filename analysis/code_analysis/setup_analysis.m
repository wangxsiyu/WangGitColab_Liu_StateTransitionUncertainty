if ~exist('versionname', 'var')
    versionname = 'ambiguity_trialnumber';
end
datadir = fullfile(W.foldernames(mfilename('fullpath')),'Temp');
datadir = fullfile(datadir, versionname);
W_lp = W_looper(datadir);
addpath(genpath('./'))
datadir = './samplegames_4';
files = W.ls(datadir);
addpath('./Code_analysis/')
%%
W.parpool
parfor fi = 1:length(files)
    findMAX(files(fi), 'Temp4');
end
W.parclose
%% load results
[re, files] = W.load('./Temp4');
MB = cellfun(@(x)x.MB.MB_RR, re);
MF = cellfun(@(x)x.Q1.Q1_RR, re);

plot([MB;MF]')
max(MB-MF)



%% just do 75%, 25%, 

versionnames = W.dir('./Temp').filename;
isplot = false;
for vi = 1:length(versionnames)
    versionname = versionnames(vi);
    %% filter 1
    files = W.ls('./*/*');
    bsnm = W.basenames(files);
    idx_main = contains(bsnm, 'main_');
    idx_draft = contains(bsnm, 'draft');
    idx_matlab = W.getext(files) == ".m";
    files = files(idx_main & ~idx_draft & idx_matlab);
    %% filter 2
    bsnm = W.basenames(files);
    bsnm = bsnm(sum(char(bsnm) == '_',2) >1);
    num = W.strs_selectbetween2patterns(bsnm, '_', '_', 1,2);
    f_a = files(~isnan(W.strs_select(num)));
    idx_fig = contains(bsnm, 'main_Figure');
    %% manual
    f_a = f_a(1:end);
    %%
    for i = 1:length(f_a)
        W.print('running: %s', W.basenames(f_a(i)));
        run(f_a(i));
    end
end
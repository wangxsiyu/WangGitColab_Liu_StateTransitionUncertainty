datadir = '../../data';
savedir = W.mkdir('../Temp');
%%
dirs = W.dir(datadir);
W.parpool;
parfor si = 1:size(dirs,1)
    files = W.dir(dirs.fullpath(si));
    tid = W.file_getprefix(files.filename) == "data";
    [tdata, tf1] = W.load(files.fullpath(tid), 'csv');
    [tefs, tf2] = W.load(files.fullpath(~tid), 'csv');
    pT = W.strs_selectbetween2patterns(W.basenames(tf1), '_pT', '_pST');
    pT = W.strs_select(pT);
    pA = W.strs_selectbetween2patterns(W.basenames(tf1), '_pA', []);
    pA = W.strs_select(pA);
    cond_gps = {find(pT == 80), find(pA == 0)};
    block_gps = reshape(0:99,50,2);
    tvers = {'ambiguity', 'uncertainty'};
    for ci = 1:length(cond_gps)
        tver = tvers{ci};
        ver1 = W.mkdir(fullfile(savedir, sprintf('%s_trialnumber', tver)));
        ver2 = W.mkdir(fullfile(savedir, sprintf('%s_reversal', tver)));
        for bi = 1:size(block_gps,2)
            data = table;
            efs = table;
            for i = 1:length(cond_gps{ci})
                di = cond_gps{ci}(i);
                trowid = ismember(tdata{1}.count_block, block_gps(:, bi));
                td = tdata{di}(trowid,:);
                td.count_block = td.count_block + 100 * (i-1);
                td = removevars(td, {'Var1', 'time_task', 'time_trial', 'state', ...
                    'p_trans', 'planet', 'is_error', 'transition', 'rewardplanet', ...
                    'randomplanet', 'p_reward_high', 'p_reward_low', 'p_ambiguity', ...
                    'p_switch_reward', 'p_switch_transition', 'ps_high_state', 'ps_low_state', 'ps_common_trans', ...
                    'ps_ambiguity', 'is_random_common0', 'spaceship'});
                td = W.tab_fill(td, 'cond_pT', pT(di));
                td = W.tab_fill(td, 'cond_pA', pA(di));
                data = W.tab_vertcat(data, td);
                te = tefs{di}(2:end, :);
                te = te(trowid,:);
                te.Var1 = [];
                efs = W.tab_vertcat(efs, te);
            end
            efs = table2array(efs);
            spikes = W.arrayfun(@(x)reshape(efs(:, x), 300, [])', 1:size(efs,2), false);
            games = W.tab_trial2game(data, [], 'count_block');
            dataset = W.struct('games', games, 'spikes', spikes);
            tname = sprintf("seed%d_session%d", si, bi);
            savename = fullfile(ver1, tname, 'dataset');   
            W.save(savename, 'data', dataset, '-v7.3');


            % compute dataset reversal
            id_rev = [find(diff(data.high_state) ~= 0)] + 1;
            win = 15;
            % distance between adjacent reversals should be win x2 (40) trials apart
            dis_pre = diff([0; id_rev]) >= win + 1;
            dis_post = diff([id_rev; size(data,1)+1]) >= win;
            id_valid = dis_pre & dis_post;
            id_rev = id_rev(id_valid);
            trialnumber_rev = W.mod0(id_rev, 300);
            trialremain_rev = 301 - trialnumber_rev;
            id_valid = trialremain_rev >= win & (trialnumber_rev >= win + 1);
            id_rev = id_rev(id_valid);
            %%
            idxsub = W.arrayfun(@(x)(x-win):(x+win-1), id_rev, false);
            games = W.tab_trial2game(data, idxsub);
            tid = vertcat(idxsub{:});
            spikes = W.arrayfun(@(x)efs(:, x), 1:size(efs,2), false);
            spikes = W.cellfun(@(x)x(tid), spikes, false);

            dataset = W.struct('games', games, 'spikes', spikes);
            tname = sprintf("seed%d_session%d", si, bi);
            savename = fullfile(ver2, tname, 'dataset');   
            W.save(savename, 'data', dataset, '-v7.3');
        end
    end
end
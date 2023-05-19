function result_svm = function_decoding(pc, games, ndim, savename)
    games = games.games;
    tpc = pc.pc;
    if ndim > 0
        tpc = W.cellfun(@(x)x(:, 1:ndim), tpc, false);
    else
        tpc = W.cellfun(@(x)x(:, (abs(ndim)+1):end), tpc, false);
    end
    ntime = length(tpc);
    ngp = 5;
    result_svm = struct('ac_choice',NaN(ngp, ntime),'se_choice',NaN(ngp, ntime),...
        'ac_highstate',NaN(ngp, ntime),'se_highstate',NaN(ngp, ntime),...
        'ac_trans',NaN(ngp, ntime),'se_trans',NaN(ngp, ntime), ...
        'ac_reward',NaN(ngp, ntime),'se_reward',NaN(ngp, ntime),...
        'ac_planet',NaN(ngp, ntime),'se_planet',NaN(ngp, ntime));
    vars ={'choice', 'highstate','trans', 'reward', 'observedplanet'};
    yy = {games.action, games.high_state, games.dominant_trans, games.reward, games.observedplanet};
    yy{3} = repmat(yy{3},1,ntime);
    for vari = 1:length(vars)
        for ti = 1:ntime % loop over time
            tx = tpc{ti};
            cc = yy{vari}(:, ti);
            tid = isnan(cc);
            if all(tid)
                [result_svm.(['ac_' varname])(:,ti), result_svm.(['se_' varname])(:,ti)] = deal(NaN);
            else
                tx = tx(~tid,:);
                cc = cc(~tid,:);
                varname = vars{vari};
                %% decode choice
                W.print('classifying %s at time point #%d/%d', varname, ti, ntime);
                SVMmd = fitcsvm(tx, cc, 'CrossVal', 'on', 'Prior', 'uniform');
                [result_svm.(['ac_' varname])(:,ti), result_svm.(['se_' varname])(:,ti)] = W.analysis_av_bygroup(SVMmd.kfoldPredict == SVMmd.Y, games.COND, unique(games.COND));
            end
        end
    end
    if exist('savename', 'var')
        W.save(savename, 'decoding_svm', result_svm);
    end
end
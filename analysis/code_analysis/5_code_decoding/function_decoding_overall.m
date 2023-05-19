function result_svm = function_decoding_overall(pc, games, ndim, savename)
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
    vars ={'observedplanet', 'choice', 'highstate','trans', 'reward'};
    yy = {games.observedplanet, games.action, games.high_state, games.dominant_trans, games.reward};
    yy{4} = repmat(yy{4},1,ntime);
    for vari = 1:length(vars)
        tx0 = vertcat(tpc{:});
        cc0 = reshape(yy{vari}, [], 1);

        tid = ~isnan(cc0);
        tx = tx0(tid,:);
        cc = cc0(tid);
        varname = vars{vari};

        W.print('classifying %s', varname);
        SVMmd = fitcsvm(tx, cc, 'CrossVal', 'on', 'Prior', 'uniform');
        is_ac = cc0 * NaN;
        is_ac(tid) = SVMmd.kfoldPredict == SVMmd.Y;
        is_ac = reshape(is_ac, [], ntime);
        [result_svm.(['ac_' varname]), result_svm.(['se_' varname])] = W.analysis_av_bygroup(is_ac, games.COND, unique(games.COND));
    end
    if exist('savename', 'var')
        W.save(savename, 'decoding_svm', result_svm);
    end
end
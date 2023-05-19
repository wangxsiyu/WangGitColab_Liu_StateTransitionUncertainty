function x1D = function_1D_projection(pc, games, savename)
    games = games.games;
    x1D = struct;
    ver_proj = 'svm';

    tpc = pc.pc;
    tpc = W.cellfun(@(x)x(:, 1:20), tpc, false);
    ntrial = size(games,1);
    ntime = length(pc.pc);
    %% projection to 1D space

    tid = games.count_trial > 0;
     tx0 = vertcat(tpc{2:end});
     cc0 = reshape(repmat(games.dominant_trans,1, 299), [], 1);

     tid = ~isnan(cc0);
     tx = tx0(tid,:);
     ty = cc0(tid);

    switch ver_proj
        case {'svm', 'svm1000', 'svm2000'}

            tmodel = fitcsvm(tx, ty, 'Prior', 'uniform');
            tw = (tmodel.SupportVectorLabels.*tmodel.Alpha)' * tmodel.SupportVectors;
            tb = tmodel.Bias;
            x1D.w_svm = tw;
            x1D.b_svm = tb;
            for ti = 1:ntime % loop over time
                xx(:, ti) = (tpc{ti} * tw'+ tb); %./sqrt(tw*tw'); % needs to double check
            end
        case 'mean' % needs to be centered
            avpos = W.analysis_av_bygroup(tx, ty, [0 1]);
            tdir = diff(avpos);
            for ti = 1:ntime % loop over time
                xx(:, ti) = tpc{ti} * tdir'/(tdir * tdir');
            end
    end
    x1D.x1D = xx; % needs to standardize
    x1D.time_at = pc.time_at;
    %%
    if exist('savename', 'var')
        W.save(savename,'x1D', x1D);
    end
end
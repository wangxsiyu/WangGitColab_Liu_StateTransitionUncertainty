function calc_value(dataset, sub, savename)
    d = preprocess_2step(dataset.games);
    d.COND = d.cond_pT * 1000 + d.cond_pA;
    gpID = unique(d.COND);

    % MLE
    for i = 1:length(gpID)
        td = d(d.COND == gpID(i),:);
        
        vs = struct;
        [~, ~, vs.Vchosen_MF ] = LL_Qlambda(sub.Q1_alpha(i), sub.Q1_sigma(i), td, 1);
        [~, ~, vs.Vchosen_MB , vs.Tchosen_MB ] = LL_MB(sub.MB_alpha(i), sub.MB_eta(i), ...
            sub.MB_sigma(i), td);
        
        nms = fieldnames(vs);
        for fi = 1:length(nms)
            d.(nms{fi})(d.COND == gpID(i),:) = vs.(nms{fi});
        end
    end

    dataset.games = d;    
    dataset.time_window = 1;
    dataset.time_at = 1:size(d.action, 2);
    W.save(savename, 'dataset', dataset);
end
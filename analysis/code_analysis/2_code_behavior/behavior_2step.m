function out = behavior_2step(d)
    d = preprocess_2step(d);
    d.COND = d.cond_pT * 1000 + d.cond_pA;
    gpID = unique(d.COND);
    out = W.analysis_tab_av_bygroup(d, 'COND', gpID, {'reward', 'correct'});
    out.gpA = mod(gpID, 1000);
    out.gpT = floor(gpID/1000);

    % MLE
    clear mlefit
    for i = 1:length(gpID)
        td = d(d.COND == gpID(i),:);
        [mlefit(i)] = MLE_2step(td);
    end
    out = W.struct_merge(out, W.structarray2struct(mlefit));

    % fit ratio
    clear mlefit
    for i = 1:length(gpID)
        td = d(d.COND == gpID(i),:);
        mlefit(i) = MLE_2step_ratio(td);
    end
    out = W.struct_merge(out, W.structarray2struct(mlefit));
end
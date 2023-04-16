function out = behavior_2step(d)
    out = W.analysis_tab_av_bybin(d, {'reward', 'correct'}, 'trialID', [0.5:1:30.5]);
end
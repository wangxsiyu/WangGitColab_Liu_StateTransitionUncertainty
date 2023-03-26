function out = be2(d)
    id3 = d.question == 3;
    id1 = d.question == 1;
    out = [];
    out.avR = W.analysis_av_bygroup(d.reward(id3), d.stepID(id3), [1:100] * 2);
    winloss = d.reward(id3);
    action = d.action(id1);
    planet = d.planet(id3);
    prob2 = W.analysis_av_bygroup(planet == 3, action, [1 2]);
    probt = [1-prob2; prob2]';
    commontrans = probt > 0.5;
    nt = length(action);
    iscommon = arrayfun(@(x)commontrans(action(x), planet(x)-1),1:nt);
    sw = action(2:end) == action(1:end-1);
    f1 = winloss(1:end-1);
    f2 = iscommon(1:end-1);
    gp = W.analysis_av_bygroup(sw, f1 * 10 + f2', [0,1,10,11]);
    out.bar = [gp([4, 2, 3, 1])];
end
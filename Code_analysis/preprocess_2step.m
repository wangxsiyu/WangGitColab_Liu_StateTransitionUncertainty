function d = preprocess_2step(d, ver)
    if ~exist('ver', 'var') || isempty(ver)
        ver = '1frame';
    end
    reward = d.reward;
    switch ver
        case '1frame'
            d.action = d.action + 1;
            d.trialID = d.tot_t;
        case '2frames'
            tid = d.stage == 1;
            d.reward = reward(find(tid)+1);
            d = d(tid,:);
            d.p_reward_low = [];
    end
    d.Var1 = [];
    d.tot_t = [];
    d.stage = [];
    d.is_done = [];
    d.t = [];
    d.observedplanet = d.obs_next;
    d.obs_next = [];
    tRP = W.cellfun(@(x) str2num(x), d.rewardplanet, false);
    d.rewardplanet = vertcat(tRP{:});

    b1 = [0; find(diff(d.trialID) < 0)] + 1;
    b2 = [b1(2:end)-1; length(d.trialID)];
    for i = 1:length(b1)
        d.blockID(b1(i):b2(i)) = i;
    end

    bID = unique(d.blockID);
    pH = W.arrayfun(@(x) nanmean(d.rewardplanet(d.blockID==x,:)), bID, false);
    pH = vertcat(pH{:});
    pH = W.arrayfun(@(x)pH(x,:), d.blockID, false);
    d.p_avRewardProb = vertcat(pH{:});

    d.planetH = d.highstate + 1; %(d.p_avRewardProb(:, 1) < d.p_avRewardProb(:, 2)) + 1;

    d.p_trans = cellfun(@(x) mean(str2num(x)), d.p_trans);
    majortrans = d.p_trans .* [1 2] + (1-d.p_trans) .* [2 1];

    trans = W.cellfun(@(x) str2num(x), d.transition, false);
    trans = vertcat(trans{:});
    d.iscommon = 0 + (trans * [1,2]' == (majortrans  * [1 2]'));

     
    d.reallandedplanet = arrayfun(@(x)trans(x, d.action(x)), 1:size(d,1))';
    d.transition = [];
    d.p_major = max(d.p_trans, 1 - d.p_trans);

    d.correct = 0 + (d.reallandedplanet == d.planetH);
end
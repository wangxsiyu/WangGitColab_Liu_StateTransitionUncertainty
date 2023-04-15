function d = preprocess_2step(d)
    tid = d.stage == 0;
    reward = d.reward;
    d = d(tid,:);
    d.reward = reward(find(tid)+1);
    d.Var1 = [];
    d.tot_t = [];
    d.stage = [];
    d.is_done = [];
    d.t = [];
    d.observedplanet = d.obs_next;
    d.obs_next = [];
    d.planetH = cellfun(@(x) str2num(x) * [1 2]', d.rewardplanet);
    d.rewardplanet = [];
    d.p_trans = cellfun(@(x) mean(str2num(x)), d.p_trans);
    d.p_reward_low = [];
    majortrans = d.p_trans .* [1 2] + (1-d.p_trans) .* [2 1];

    trans = W.cellfun(@(x) str2num(x), d.transition, false);
    trans = vertcat(trans{:});
    d.iscommon = 0 + (trans * [1,2]' == (majortrans  * [1 2]'));

    d.reallandedplanet = arrayfun(@(x)trans(x, d.action(x)), 1:size(d,1))';
    d.transition = [];
    d.p_major = max(d.p_trans, 1 - d.p_trans);

    d.correct = 0 + (d.reallandedplanet == d.planetH);
end
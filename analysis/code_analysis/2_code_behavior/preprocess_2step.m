function d = preprocess_2step(d)
    fillernan = nan(size(d,1),1);
    d.action = d.action + 1;
    d.high_state = d.high_state + 1;
    d.observedplanet = [d.obs(:,2:end), fillernan];
    % correct choice
    correctaction = d.high_state;
    correctaction(d.dominant_trans == 2,:) = 3 - correctaction(d.dominant_trans == 2,:);
    d.correct = correctaction == d.action;
end
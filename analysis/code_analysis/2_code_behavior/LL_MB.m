
function [LL, Q1out, Qchosen, Tchosen] = LL_MB(alpha, eta, sigma, d)
    nblock = size(d, 1);
    ntrial = size(d.action, 2);
    myL = nan(nblock, ntrial);
    Qchosen = myL;
    Tchosen = myL;
    Q1out = {myL, myL};
    for bi = 1:nblock
        Q2 = [0, 0];
        pT = ones(2,2) * 0.5;
        for i = 1:ntrial
            Q1 = pT * Q2';
            Q1out{1}(bi, i) = Q1(1);
            Q1out{2}(bi, i) = Q1(2);
            p = W.softmax_binary(Q1(1), Q1(2), sigma);
            action = d.action(bi, i);
            Qchosen(bi, i) = Q1(action);
            Tchosen(bi, i) = pT(action, action);
            myL(bi, i) = p(action);
            planet = d.observedplanet(bi, i);
            if ~isnan(planet)
                r = d.reward(bi, i);
                Q2(planet) = (1 - alpha) * Q2(planet) + alpha * r;
                pT(action, planet) = pT(action, planet) + eta * (1-pT(action, planet));
                pT(action, 3-planet) = pT(action, 3-planet) + eta * (0-pT(action, 3-planet));
            end
        end
    end
    myL = log(myL);
    LL = mean(myL, 'all', 'omitnan');
end
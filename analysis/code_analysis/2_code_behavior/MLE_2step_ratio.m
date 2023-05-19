function out = MLE_2step_ratio(d)
    fmcopt = optimoptions(@fmincon,'Algorithm','sqp');
    fit1 = fit_mixture(d, fmcopt);
    out = fit1;
end

function out = fit_mixture(d, fmcopt)
    %   alpha (MB), eta (MB), alpha (MF), sigma, ratio
    X0 = [0.5, 0.5, 0.5, 1, 0.5];
    LB = [0, 0, 0, 0, 0];
    UB = [1, 1, 1, 9, 1];
    func = @(x)-LL_mixture(x(1), x(2), x(3), x(4), x(5), d);
    [xfit, LL] = fmincon(func, X0, [],[],[],[], LB, UB, [], fmcopt);
    out.mix_MB_alpha = xfit(1);
    out.mix_MB_eta = xfit(2);
    out.mix_MF_alpha = xfit(3);
    out.mix_sigma = xfit(4);
    out.mix_ratioMB = xfit(5);
    out.mix_LL = -LL;
end

function LL = LL_mixture(alpha_MB, eta, alpha_MF, sigma, ratio, d)
    nblock = size(d, 1);
    ntrial = size(d.action, 2);
    myL = nan(nblock, ntrial);
    for bi = 1:nblock
        Q2MB = [0, 0];
        pT = ones(2,2) * 0.5;
        Q1MF = [0, 0];
        for i = 1:ntrial
            Q1MB = W.horz(pT * Q2MB');
            Q1 = Q1MB * ratio + (1-ratio) * Q1MF;
            p = W.softmax_binary(Q1(1), Q1(2), sigma);
            action = d.action(bi, i);
            myL(bi, i) = p(action);
            r = d.reward(bi, i);
            Q1MF(action) = (1 - alpha_MF) * Q1MF(action) + alpha_MF * r;
            planet = d.observedplanet(bi, i);
            if ~isnan(planet)
                Q2MB(planet) = (1 - alpha_MB) * Q2MB(planet) + alpha_MB * r;
                pT(action, planet) = pT(action, planet) + eta * (1-pT(action, planet));
                pT(action, 3-planet) = pT(action, 3-planet) + eta * (0-pT(action, 3-planet));
            end
        end
    end
    myL = log(myL);
    LL = mean(myL, 'all', 'omitnan');
end

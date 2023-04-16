function out = MLE_2step(d)
    out = fit_Qlambda(d, 1);
%     fit2 = fit_Qlambda(d, 0);
%     out = W.struct_merge(out, fit2);
    fit2 = fit_MB(d);
    out = W.struct_merge(out, fit2);
end

function out = fit_MB(d)
    %   alpha, eta, sigma
    X0 = [0, 0, 1];
    LB = [0, 0, 0];
    UB = [1, 1, 99];
    func = @(x)-LL_MB(x(1), x(2), x(3), d);
    [xfit, LL] = fmincon(func, X0, [],[],[],[], LB, UB);
    out.MB_alpha = xfit(1);
    out.MB_eta = xfit(2);
    out.MB_sigma = xfit(3);
    out.MB_LL = -LL;
end

function LL = LL_MB(alpha, eta, sigma, d)
    ntrial = size(d, 1);
    myL = nan(1, ntrial);
    for i = 1:ntrial
        if i == 1 || d.trialID(i) == 1 || (i > 1 && d.trialID(i) < d.trialID(i-1))
            Q2 = [0,0];
            pT = ones(2,2) * 0.5;
        end
        Q1 = pT * Q2';
        p = mysoftmax (Q1(1), Q1(2), sigma);
        action = d.action(i);
        planet = d.observedplanet(i);
        myL(i) = p(action);
        r = d.reward(i);
        Q2(planet) = (1 - alpha) * Q2(planet) + alpha * r;
        pT(action, planet) = pT(action, planet) + eta * (1-pT(action, planet));
        pT(action, 3-planet) = pT(action, 3-planet) + eta * (0-pT(action, 3-planet));
    end
    myL = log(myL);
    LL = mean(myL, 'all', 'omitnan');
end

function out = fit_Qlambda(d, lambda)
    %   alpha, sigma
    X0 = [0, 1];
    LB = [0, 0];
    UB = [1, 99];
    func = @(x)-LL_Qlambda(x(1), x(2), d, lambda);
    [xfit, LL] = fmincon(func, X0, [],[],[],[], LB, UB);
    out.(sprintf('Q%d_alpha', lambda)) = xfit(1);
    out.(sprintf('Q%d_sigma', lambda)) = xfit(2);
    out.(sprintf('Q%d_LL', lambda)) = -LL;
end

function LL = LL_Qlambda(alpha, sigma, d, lambda)
    ntrial = size(d, 1);
    myL = nan(1, ntrial);
    for i = 1:ntrial 
        if i == 1 || d.trialID(i) == 1 || (i > 1 && d.trialID(i) < d.trialID(i-1))
            Q1 = [0,0];
            Q2 = 0;
        end
        p = mysoftmax (Q1(1), Q1(2), sigma);
        action = d.action(i);
        myL(i) = p(action);
        r = d.reward(i);
        Q1(action) = (1 - alpha) * Q1(action) + alpha * (Q2 + lambda*(r - Q2));
        Q2 = (1 - alpha) * Q2 + alpha * r;
    end
    myL = log(myL);
    LL = mean(myL, 'all', 'omitnan');
end

function p = mysoftmax(v1, v2, sigma)
   dq = v2 - v1;
   p = 1./(1 + exp(-dq/sigma));
   p = [1-p, p];
end
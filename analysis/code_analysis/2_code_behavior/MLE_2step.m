function [out] = MLE_2step(d)
    fmcopt = optimoptions(@fmincon,'Algorithm','sqp');
    fit1 = fit_Qlambda(d, 1, fmcopt);
%     fit2 = fit_Qlambda(d, 0);
%     out = W.struct_merge(out, fit2);
    fit2 = fit_MB(d, fmcopt);
    out = W.struct_merge(fit1, fit2);
end

function out = fit_MB(d, fmcopt)
    %   alpha, eta, sigma
    X0 = [0.5, 0.5, 1];
    LB = [0, 0, 0];
    UB = [1, 1, 9];
    func = @(x)-LL_MB(x(1), x(2), x(3), d);
    [xfit, LL] = fmincon(func, X0, [],[],[],[], LB, UB, [], fmcopt);
    out.MB_alpha = xfit(1);
    out.MB_eta = xfit(2);
    out.MB_sigma = xfit(3);
    out.MB_LL = -LL;
end


function out = fit_Qlambda(d, lambda, fmcopt)
    %   alpha, sigma
    X0 = [0.5, 1];
    LB = [0, 0];
    UB = [1, 9];
    func = @(x)-LL_Qlambda(x(1), x(2), d, lambda);
    [xfit, LL] = fmincon(func, X0, [],[],[],[], LB, UB, [], fmcopt);
    out.(sprintf('Q%d_alpha', lambda)) = xfit(1);
    out.(sprintf('Q%d_sigma', lambda)) = xfit(2);
    out.(sprintf('Q%d_LL', lambda)) = -LL;
end

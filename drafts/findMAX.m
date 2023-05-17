function findMAX(filename, savedir)
    tname = W.basenames(filename);
    tname = W.file_prefix(W.file_deprefix(tname, 0), 'xfit');
    tsavename = fullfile(W.mkdir(savedir), tname);
    tsavename = strcat(tsavename, '.mat');
    if exist(tsavename, 'file') && false
        W.print('file exists, skip: %s', W.basenames(tsavename));
    else
        d = W.load(filename, 'csv');
        d = preprocess_game2step(d);
        out = struct;
        tic
        out.Q1 = findMAX_Q1(d);
        toc
        tic
        out.MB = findMAX_MB(d);
        toc
        W.save(tsavename, 'xfit', out);
    end
end
function out = findMAX_Q1(d, lambda)
    if ~exist('lambda', 'var')
        lambda = 1;
    end
    %   alpha, sigma
%     X0 = [0.5];
%     LB = [0];
%     UB = [1];
    X0 = [1 .1];
    LB = [0 0];
    UB = [1 1];
    func = @(x)-RR_Qlambda(x(1), x(2), d, lambda);
    [xfit, LL] = fmincon(func, X0, [],[],[],[], LB, UB);
%     xfit = NaN;
%     LL = -Inf;
%     xs = 0:0.1:1;
%     for xi = 1:length(xs)
%         tR = RR_Qlambda(xs(xi), 0.5, d, 1);
%         if tR > LL 
%             LL = tR;
%             xfit = xs(xi);
%         end
%     end
    
    out.(sprintf('Q%d_alpha', lambda)) = xfit(1);
    out.(sprintf('Q%d_sigma', lambda)) = xfit(2);
    out.(sprintf('Q%d_RR', lambda)) = -LL;
end
function R = RR_Qlambda(alpha, sigma, d, lambda)
    ntrial = size(d, 1);
    R = [];
    for repi = 1:1
        for i = 1:ntrial
            if i == 1 || d.trialID(i) == 1 || (i > 1 && d.trialID(i) < d.trialID(i-1))
                Q1 = [0,0];
                Q2 = 0;
            end
            p = mysoftmax (Q1(1), Q1(2), sigma);
            action = sampleaction(p);
            planet = d.trans(i, action);
            r = d.rewardplanet(i, planet);
            if ~isnan(d.randomplanet(i))
                planet = d.randomplanet(i) + 1;
            end
            Q1(action) = (1 - alpha) * Q1(action) + alpha * (Q2 + lambda*(r - Q2));
            Q2 = (1 - alpha) * Q2 + alpha * r;
            R(repi, i) = r;
        end
    end
    R = mean(R, "all");
end
function out = findMAX_MB(d)
    X0 = [1, 1, .1];
    LB = [0, 0, 0];
    UB = [1, 1, 1];
%     X0 = [0.5, 0.5];
%     LB = [0, 0];
%     UB = [1, 1];
    func = @(x)-RR_MB(x(1), x(2), x(3), d);
    [xfit, LL] = fmincon(func, X0, [],[],[],[], LB, UB);
%     xfit = NaN(1,2);
%     LL = -Inf;
%     xs = 0:0.1:1;
%     ys = 0:0.1:1;
%     for xi = 1:length(xs)
%         for yi = 1:length(ys)
%             tR = RR_MB(xs(xi), ys(yi), 0.5, d);
%             if tR > LL
%                 LL = tR;
%                 xfit = [xs(xi), ys(yi)];
%             end
%         end
%     end


    out.MB_alpha = xfit(1);
    out.MB_eta = xfit(2);
    out.MB_sigma = xfit(3);
    out.MB_RR = -LL;
end
function R = RR_MB(alpha, eta, sigma, d)
    ntrial = size(d, 1);
    R = [];
    for repi = 1:1
        for i = 1:ntrial
            if i == 1 || d.trialID(i) == 1 || (i > 1 && d.trialID(i) < d.trialID(i-1))
                Q2 = [0,0];
                pT = ones(2,2) * 0.5;
            end
            Q1 = pT * Q2';
            p = mysoftmax (Q1(1), Q1(2), sigma);
            action = sampleaction(p);
            planet = d.trans(i, action);
            r = d.rewardplanet(i, planet);
            if ~isnan(d.randomplanet(i))
                planet = d.randomplanet(i) + 1;
            end
            Q2(planet) = (1 - alpha) * Q2(planet) + alpha * r;
            pT(action, planet) = pT(action, planet) + eta * (1-pT(action, planet));
            pT(action, 3-planet) = pT(action, 3-planet) + eta * (0-pT(action, 3-planet));
            R(repi, i) = r;
        end
    end
    R = mean(R, "all");
end
function p = mysoftmax(v1, v2, sigma)
   dq = v2 - v1;
   p = 1./(1 + exp(-dq/sigma));
   p = [1-p, p];
end
function c = sampleaction(p)
    c = sum(rand > cumsum(p)) + 1;
end
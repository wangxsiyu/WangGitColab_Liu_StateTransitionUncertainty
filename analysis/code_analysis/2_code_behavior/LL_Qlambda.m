
function [LL, Q1out, Qchosen] = LL_Qlambda(alpha, sigma, d, lambda)
    nblock = size(d, 1);
    ntrial = size(d.action, 2);
    myL = nan(nblock, ntrial);
    Qchosen = myL;
    Q1out = {myL, myL};
    for bi = 1:nblock
        Q1 = [0,0];
        Q2 = 0;
        for i = 1:ntrial
            Q1out{1}(bi, i) = Q1(1);
            Q1out{2}(bi, i) = Q1(2);
            p = W.softmax_binary(Q1(1), Q1(2), sigma);
            action = d.action(bi, i);
            Qchosen(bi, i) = Q1(action);
            myL(bi, i) = p(action);
            r = d.reward(bi, i);
            Q1(action) = (1 - alpha) * Q1(action) + alpha * (Q2 + lambda*(r - Q2));
            Q2 = (1 - alpha) * Q2 + alpha * r;
        end
    end
    myL = log(myL);
    LL = mean(myL, 'all', 'omitnan');
end

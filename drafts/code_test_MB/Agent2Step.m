classdef Agent2Step < handle
    properties
    end
    methods(Static)
        function d = simu_MB(alpha, eta, sigma, d)
            ntrial = size(d, 1);
            for i = 1:ntrial
                if i == 1 || d.trialID(i) == 1 || (i > 1 && d.trialID(i) < d.trialID(i-1))
                    Q2 = [0,0];
                    pT = ones(2,2) * 0.5;
                end
                Q1 = pT * Q2';
                p = Agent2Step.mysoftmax (Q1(1), Q1(2), sigma);
                if rand < p(1)
                    action = 1;
                else
                    action = 2;
                end
                if rand < d.p_trans(i)
                    planet = action;
                else
                    planet = 3-action;
                end
                r = d.rewardplanet(i, planet);
                Q2(planet) = (1 - alpha) * Q2(planet) + alpha * r;
                pT(action, planet) = pT(action, planet) + eta * (1-pT(action, planet));
                pT(action, 3-planet) = pT(action, 3-planet) + eta * (0-pT(action, 3-planet));
            
                d.action(i) = action;
                d.reallandedplanet(i) = planet;
                d.observedplanet(i) = planet;
                d.reward(i) = r;
            end
        end
        function p = mysoftmax(v1, v2, sigma)
           dq = v2 - v1;
           p = 1./(1 + exp(-dq/sigma));
           p = [1-p, p];
        end
    end
end
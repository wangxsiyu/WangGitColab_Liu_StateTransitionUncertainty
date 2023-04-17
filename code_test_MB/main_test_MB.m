d = W.readtable('./data_pR100_pSR0.0_pT80_pST0.0_PST0T_pA0.csv');
%%
addpath('../Code_analysis/');
d = preprocess_2step(d);
%%
xfit = MLE_2step(d);
%%
d1 = Agent2Step.simu_MB(xfit.MB_alpha, 0.5, xfit.MB_sigma, d);
%%
xfit2 = MLE_2step(d1);
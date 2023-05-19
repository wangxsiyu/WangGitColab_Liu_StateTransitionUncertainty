run('../setup_plot.m')
%%
pcs = W_lp.load('pca');
data = W_lp.load('data_cleaned');
decode = W_lp.load('decoding_overall');
%%
for si = 1:length(pcs)
    %%
    plt.figure(2,3, 'is_title', 'all', 'matrix_hole', [1 1 1; 1 1 0]);
    pc = pcs{si}.pc;
    pc = W.cellfun(@(x)x(:,1:3), pc, false);
    d = data{si}.games;
    %
    cs = unique(d.COND);
    for axi = 1:5
        plt.ax(axi);
        id_cond = d.COND == cs(axi);
        cond = [d.dominant_trans, d.high_state(:,1)];
        conds = unique(cond, 'rows');

        for ci = 1:length(conds)
            tid = all(conds(ci,:) == cond, 2) & id_cond;
            tpc = W.cellfun(@(x)mean(x(tid,:)), pc, false);
            tpc = vertcat(tpc{:});
            traj{ci} = tpc;
        end


        plt.setfig_ax('xlabel', 'PC1', 'ylabel', 'PC2', 'zlabel', 'PC3', ...
            'title', sprintf('pT = %.0f, pA = %.0f', unique(d.cond_pT(id_cond)), unique(d.cond_pA(id_cond))));
        if axi == 5
            plt.setfig_ax('legend', ...
                {'ips, high = L, opt = L', 'ips, high = R, opt = R', 'con, high = L, opt = R', 'con, high = R, opt = L'});
            plt.setfig_ax('legloc', 'SEO');
        end
        grid on
        cols = {'AZred', 'AZred', 'AZblue', 'AZblue'};
        shps = {'-','--','-', '--'};
        for ci = 1:length(conds)
            tcol = cols{ci};
            plt.plot3D(traj{ci}, 'shape', shps{ci}, 'color', {strcat(tcol, '20'), tcol}, ...
                'gradient_color', true);
        end
    end
    plt.unify_lims();
    
    plt.update('traj1');
    %%
    plt.figure(2,3, 'is_title', 'all', 'matrix_hole', [1 1 1; 1 1 0]);
    pc = pcs{si}.pc;
    pc = W.cellfun(@(x)x(:,1:3), pc, false);
    d = data{si}.games;
    %
    cs = unique(d.COND);
    for axi = 1:5
        plt.ax(axi);
        id_cond = d.COND == cs(axi);
        cond = [d.dominant_trans];
        conds = unique(cond, 'rows');

        traj = {};
        for ci = 1:length(conds)
            tid = all(conds(ci,:) == cond, 2) & id_cond;
            tpc = W.cellfun(@(x)mean(x(tid,:)), pc, false);
            tpc = vertcat(tpc{:});
            traj{ci} = tpc;
        end


        plt.setfig_ax('xlabel', 'PC1', 'ylabel', 'PC2', 'zlabel', 'PC3', ...
            'title', sprintf('pT = %.0f, pA = %.0f', unique(d.cond_pT(id_cond)), unique(d.cond_pA(id_cond))));
        if axi == 5
            plt.setfig_ax('legend', ...
                {'ips', 'ips, high = R, opt = R', 'con, high = L, opt = R', 'con, high = R, opt = L'});
            plt.setfig_ax('legloc', 'SEO');
        end
        grid on
        cols = {'AZred', 'AZblue'};
        shps = {'-','-'};
        for ci = 1:length(conds)
            tcol = cols{ci};
            plt.plot3D(traj{ci}(2:50,:), 'shape', shps{ci}, 'color', {strcat(tcol, '20'), tcol}, ...
                'gradient_color', true);
        end
    end
    plt.unify_lims();
    
    plt.update('traj1');
end
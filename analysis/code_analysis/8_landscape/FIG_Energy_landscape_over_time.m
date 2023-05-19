function FIG_Energy_landscape_over_time(plt, x1D, time_at, EL, games, animal)
    %%
    conds = unique(games{1}.COND);
    avtraj = {[],[]};
    for ti = 1:length(x1D)
        for ci = 1:5
            tid = conds(ci) == games{ti}.COND;
            tt = W.analysis_av_bygroup(x1D{ti}(tid,:), games{ti}.dominant_trans(tid,:), [1,2]);
            avtraj{1,ci}(ti,:) = tt(1,:);
            avtraj{2,ci}(ti,:) = tt(2,:);
        end
    end
    %%
    time_EL = EL{1}.time_EL;
    x_EL = EL{1}.x_EL;
%     time_at = x1D{1}.time_at;
    EL_merged = W.cellfun(@(x)x.EL_cue, EL, false);
    EL_merged = vertcat(EL_merged{:});
    %% 
    avEL = cell(1, size(EL_merged,2));
    avTJ = cell(1, size(EL_merged,2));
    for i = 1:size(EL_merged,2)
        % average between monkeys
        avEL{i} = W.cell_mean(EL_merged(:,i));
        % average traj
        avTJ{i} = [W.avse(avtraj{1, i}); W.avse(avtraj{2, i})];
    end
    %% development
    plt.figure(2,3,'is_title', 1, 'gapW_custom', [0 1 1 1] * 50, 'matrix_hole', [1 1 1; 1 1 0]);
    for i = 1:5
        plt.setfig_ax('xlabel','trial number',...
            'ylabel', 'position', ...
        'xtickangle', 0);
        pcolor(time_EL, x_EL, avEL{i}');
        shading interp 

%         set(gca,'color',0*[1 1 1]);
%         ylim([-2.5 2.5])
        tpos = get(plt.fig.axes(plt.fig.axi), 'position');
        colorbar;
        caxis(quantile(reshape(avEL{i},[],1), [0.025 0.975]));
        set(plt.fig.axes(plt.fig.axi), 'position', tpos);
        hold on;
        plt.plot(time_at, avTJ{i}, ...
            [], 'line', 'color', plt.custom_vars.col_MBMF);
        plt.new;
    end
    plt.update('development');
end
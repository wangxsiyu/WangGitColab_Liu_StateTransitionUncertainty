function FIG_Energy_landscape_by_cue_new(plt, EL, sub, timeslice)
    time_EL = EL{1}.time_EL;
    x_EL = EL{1}.x_EL;
    %% compile curves by cue
    EL_cue = cell(1, length(EL));
    grad_cue = cell(1, length(EL));
    for si = 1:length(EL)
        if W.is_stringorchar(timeslice)
            ttmslice = sub.midRT_REJECT(si) * 1000;
        else
            ttmslice = timeslice;
        end
        idxt = dsearchn(time_EL', ttmslice);
        tEL = W.cellfun(@(t)t(idxt,:), EL{si}.EL_cue, false);
        tG = W.cellfun(@(t)t(idxt,:), EL{si}.grad_cue, false);
        EL_cue{si} = vertcat(tEL{:});
        grad_cue{si} = vertcat(tG{:});
    end
    %% average between monkeys
    mks = unique(sub.animal);
    avEL = {};
    seEL = {};
    pa = [];
    for i = 1:length(mks)
        [avEL{i}, seEL{i}] = W.cell_avse(EL_cue);
        [avgrad{i}, segrad{i}] = W.cell_avse(grad_cue);
%         [avEL{i}] = W.calc_EL_integral(avgrad{i}, x_EL, 0);
        % compute pa
%         pa(i,:) = mean(sub.avCHOICE_byCONDITION(sub.animal == mks(i),:));
    end
    pa = EL{1}.conds_cue;
    si = 1;
    %% development
    plt.figure(1,1,'is_title', 1, 'gapW_custom', [0 1] * 100);
%     plt.setfig('title', W.str2cell(W.file_prefix(mks, 'Monkey',' ')));
    for i = 1:1
        condcolors = plt.custom_vars.color_cond;
        leg = arrayfun(@(x)sprintf("pT = %.0f, pA = %.0f", floor(x/1000), mod(x, 1000)), pa(i));
        plt.setfig_ax('xlabel', 'position', 'ylabel', 'V', ...
            'legloc','eastoutside',...
            'legend', leg, 'xlim', [-5,5]);
        plt.plot(x_EL, avEL{i},seEL{i},'shade', 'color', condcolors);
%         plt.plot(x_EL, EL_cue{si}, [],'line', 'color', condcolors);
        plt.new;
    end
    plt.update([],'  ');
    plt.save(sprintf('energy landscape by cue at %sms', W.string(timeslice)));
end
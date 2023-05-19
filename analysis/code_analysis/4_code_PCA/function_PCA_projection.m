function pc = function_PCA_projection(data, savename)
    if exist('savename', 'var') && exist(savename, 'file')
        pc = W.load(savename);
        return;
    end
    %%
    pc = struct;
    [tpc, pc.pca_r2, pc.coeff, pc.mu] = ...
            W.neuro_PCA(data.spikes, []);
    pc.pc = tpc; %W.cellfun(@(x)x(:, 1:20), tpc);
    pc.time_at = data.time_at;
    if exist('savename', 'var')
        W.save(savename, 'pc', pc);
    end
end


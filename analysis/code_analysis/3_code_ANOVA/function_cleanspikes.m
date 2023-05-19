function data = function_cleanspikes(data, varargin)
    savenames = varargin{end};
    savenames = W.string(savenames);
    data.time_window = 1;
    data.time_at = 1:300;
    [data, spikes_cleaning_info] = W.clean_spikes(data,varargin{1:end-1});
    W.save(savenames(1), 'data', data);
    W.save(savenames(2), 'spikes_cleaning_info', spikes_cleaning_info);
end
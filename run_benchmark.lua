--[[
    Train and evaluate a series of different rnn modules on three datasets (Shakespear, Linux kernel and Wikipedia).
]]


require 'paths'
require 'torch'
require 'gnuplot'

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Utility functions
--------------------------------------------------------------------------------

local function load_log(filename)
    local data = {}
    if not paths.filep(filename) then
        error(('\nFile %s does not exist'):format(filename))
    end
    local f = io.open(filename, 'r')
    for line in f:lines() do
        table.insert(data, line)
    end
    f:close()
    table.remove(data,1)
    local out = {}
    for i=1, #data do
        local str = data[i]:split('\t')
        for j=1, #str do
            if out[j] then
                table.insert(out[j], tonumber(str[j]))
            else
                out[j] = {tonumber(str[j])}
            end
        end
    end
    return out
end

------------------------------------------------------------------------------------------------------------

local function parse_epoch_loss_log(filename)
    local data = load_log(filename)
    local train_loss = data[1]
    local test_loss = data[2]
    return train_loss, test_loss
end

------------------------------------------------------------------------------------------------------------

local function parse_epoch_info_log(filename)
    local data = load_log(filename)
    local forward_time = data[1]
    local backward_time = data[2]
    local batch_time = data[3]
    local memory_mb = data[4]
    return forward_time, backward_time, batch_time, memory_mb
end

------------------------------------------------------------------------------------------------------------

local function get_sorted_keys_table(tableA)
    local keys = {}
    for k,v in pairs(tableA) do
        table.insert(keys, k)
    end
    table.sort(keys)
    return keys
end

------------------------------------------------------------------------------------------------------------

local function plot_graph(data, filename, title, x_label, y_label)
    local save_dir = 'data/results/'
    gnuplot.pngfigure(paths.concat(save_dir, filename))
    gnuplot.plot(data)
    gnuplot.xlabel(x_label)
    gnuplot.ylabel(y_label)
    gnuplot.title(title)
    gnuplot.grid(true)
    gnuplot.axis('auto')
    gnuplot.plotflush()
end

------------------------------------------------------------------------------------------------------------

local function concat_table(tableA, tableB)
    for i=1, #tableB do
        table.insert(tableA, tableB[i])
    end
    return tableA
end

------------------------------------------------------------------------------------------------------------

local function mean_table(tableA)
    local mean = 0
    for k, value in pairs(tableA) do
        mean = mean + value
    end
    return mean / #tableA
end

------------------------------------------------------------------------------------------------------------

local function convert_data(hiddendim_data, str_suffix, plt_format)
    local model_data = {}
    local str_suffix = str_suffix or ''
    local dimensions = get_sorted_keys_table(hiddendim_data)
    for _, dim in ipairs(dimensions) do
        data = hiddendim_data[dim]
        for model, mdata in pairs(data) do
            local mean_hiddendim = mean_table(mdata)
            if model_data[model] then
                table.insert(model_data[model], mean_hiddendim)
            else
                model_data[model] = {mean_hiddendim}
            end
        end
    end
    local out = {}
    for model, data in pairs(model_data) do
        table.insert(out, {model .. str_suffix, torch.Tensor(dimensions), torch.Tensor(data), plt_format or '-'})
    end
    return out
end


--------------------------------------------------------------------------------
-- Train all networks on all datasets
--------------------------------------------------------------------------------

g_skip_train = false
local hidden_dimensions = {64, 128, 256, 512, 1024}
local datasets = {'shakespear', 'linux', 'wikipedia'}

local hidden_dimensions_info = {}
for i, dim in ipairs(hidden_dimensions) do
    g_hidden_dimension_size = dim  -- global variable indicating
                                                  -- the rnn layer dimension for
                                                  -- train
    local dataset_configs = {}
    for _, dataset in pairs(datasets) do
        g_dataset = dataset  -- global variable to select the dataset type
        local configs_dataset = paths.dofile('scripts/train_all.lua')
        dataset_configs[g_dataset] = configs_dataset
    end
    hidden_dimensions_info[dim] = dataset_configs
end


--------------------------------------------------------------------------------
-- Compute plots for each dataset
--------------------------------------------------------------------------------

local forward_stats = {}
local backward_stats = {}
local batch_stats = {}
local gpu_memory_stats = {}
local dimensions = get_sorted_keys_table(hidden_dimensions_info)
for _, dim in ipairs(dimensions) do
    local configs_dataset = hidden_dimensions_info[dim]
    local dim_stats = {}
    local loss_epoch_stats = {}

    --[[ fetch data from all trained datasets ]]--
    local forward_data = {}
    local backward_data = {}
    local batch_data = {}
    local memory_data = {}
    for dataset, configs in pairs(configs_dataset) do
        local loss_data = {}
        for model, config in pairs(configs) do
            local log_loss_filename = ('./data/exp/%s/%s/epoch_loss.log'):format(dataset, config.expID)
            local log_info_filename = ('./data/exp/%s/%s/epoch_info.log'):format(dataset, config.expID)

            print(('\nProcessing experiment log: %s/%s'):format(dataset, config.expID))
            local train_loss, test_loss = parse_epoch_info_log(log_loss_filename)
            local forward_time, backward_time, batch_time, memory_mb = parse_epoch_info_log(log_info_filename)

            if forward_data[model] then
                table.insert(forward_data[model],  mean_table(forward_time) * 1000)
                table.insert(backward_data[model], mean_table(backward_time) * 1000)
                table.insert(batch_data[model],    mean_table(batch_time) * 1000)
                table.insert(memory_data[model],   mean_table(memory_mb))
            else
                forward_data[model]  = {mean_table(forward_time) * 1000}
                backward_data[model] = {mean_table(backward_time) * 1000}
                batch_data[model]    = {mean_table(batch_time) * 1000}
                memory_data[model]   = {mean_table(memory_mb)}
            end

            table.insert(loss_data, {model, torch.range(1, #test_loss), torch.FloatTensor(test_loss), '-'})
            --table.insert(loss_data, {model .. ' (train)', torch.range(1, #train_loss), torch.FloatTensor(train_loss), '+-'})
        end

        -- loss per epoch graph
        plot_graph(loss_data,
                ('loss_%s_%d.png'):format(dataset, dim),
                'Train/Test Loss per Epoch ' .. ('(Dim %d)'):format(dim),
                'Epoch',
                'Loss')
    end

    forward_stats[dim] = forward_data
    backward_stats[dim] = backward_data
    batch_stats[dim] = batch_data
    gpu_memory_stats[dim] = memory_data
end

--[[ plot graphs ]]--
-- batch time vs hidden dimensions
plot_graph(convert_data(batch_stats),
           'speed_vs_dimension.png',
           'Speed vs Hidden Dimension for Depth 2',
           'Hidden Dimension',
           'Speed (ms/batch)')

-- gpu memory usage vs hidden dimensions
plot_graph(convert_data(gpu_memory_stats),
           'memory_vs_dimension.png',
           'GPU Memory Usage vs Hidden Dimension for Depth 2',
           'Hidden Dimension',
           'Memory (MB)')

-- forward/backward time vs hidden dimensions
plot_graph(concat_table(convert_data(forward_stats, ' (fw)'),
                        convert_data(backward_stats, ' (bw)', '+-')),
           'fw_bw_vs_dimension.png',
           'Forward(fw) + Backward(bw) Speed vs Hidden Dimension for Depth 2',
           'Hidden Dimension',
           'Speed (ms/batch)')

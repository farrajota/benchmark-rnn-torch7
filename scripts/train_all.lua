--[[
    Runs all the scripts in this directory used for training a RNN network.
]]


--[[ Load network trainer script ]]--
local train_net = paths.dofile('train_rnn_network.lua')

--[[ Load train configurations ]]--
local configs = paths.dofile('train_configs.lua')

--[[ Networks ]]--
local networks_configs = {  -- modelID = model_name_alias
    -- Vanilla
    rnn_vanilla = 'rnn-vanilla',
    lstm_vanilla = 'lstm-vanilla',

    -- rnn (Element-Research)
    rnn_rnn = 'rnn-rnn',
    lstm_rnn = 'lstm-rnn',
    fastlstm_rnn = 'lstm(fast)-rnn',
    gru_rnn = 'gru-rnn',

    -- rnnlib (facebook)
    rnn_rnnlib = 'rnn-rnnlib',
    lstm_rnnlib = 'lstm-rnnlib',
    gru_rnnlib = 'gru-rnnlib',

    -- cudnn
    rnnrelu_cudnn = 'rnn(relu)-cudnn',
    rnntanh_cudnn = 'rnn(tanh)-cudnn',
    lstm_cudnn = 'lstm-cudnn',
    blstm_cudnn = 'blstm-cudnn',
    gru_cudnn = 'gru-cudnn',
}

local output = {}
if g_skip_train then
    --[[ Return the configurations for all networks ]]--
    for k, v in pairs(networks_configs) do
        local configs = paths.dofile('train_configs.lua')
        configs.expID = v .. '-' .. g_hidden_dimension_size
        configs.model = k
        configs.dataset = g_dataset
        configs.rnn_size = g_hidden_dimension_size
        output[k] = configs
    end
else
    --[[ Train networks ]]--
    for k, v in pairs(networks_configs) do
        output[k] = train_net(configs, v, k)
    end
end

--[[ Return the configurations for all trained networks ]]--
return output

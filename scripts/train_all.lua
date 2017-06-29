--[[
    Runs all the scripts in this directory used for training a RNN network.
]]


--[[ Load network trainer script ]]--
local train_net = paths.dofile('train_rnn_network.lua')

--[[ Load train configurations ]]--
local configs = paths.dofile('train_configs.lua')

--[[ Train networks ]]--
local networks_configs = {
    -- Vanilla
    rnn_vanilla = train_net(configs, 'rnn-vanilla', 'rnn_vanilla'),
    lstm_vanilla = train_net(configs, 'lstm-vanilla', 'lstm_vanilla'),

    -- rnn (Element-Research)

    -- rnnlib (facebook)

    -- cudnn
}

--[[ Return the configurations for all networks ]]--
return networks_configs
--[[
    Script to train a RNN network.
]]


local function exec_command(command)
    print('\n')
    print('Executing command: ' .. command)
    print('\n')
    os.execute(command)
end

------------------------------------------------------------------------------------------------------------

local function train_net(configs, expID, modelID)
    --[[ configurations ]]--
    local configs = paths.dofile('train_configs.lua')
    configs.expID = expID .. '-' .. g_hidden_dimension_size
    configs.model = modelID
    configs.dataset = g_dataset
    configs.rnn_size = g_hidden_dimension_size


    --[[ Parse the configurations ]]--
    local str_args = ''
    for k, v in pairs(configs) do
        str_args = str_args .. ('-%s %s '):format(k, v)
    end


    --[[ Start training the network ]]--
    exec_command(('th train.lua %s'):format(str_args))


    --[[ output the configurations ]]--
    return configs
end

------------------------------------------------------------------------------------------------------------

return train_net
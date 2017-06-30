--[[
    Hyperparameter experimentation using the ADAM optimizer for training a network.

    Note: run this file from the root dir of the repo.

          Example: th tests/test_hyperparameters_adam.lua
]]


require 'paths'
require 'torch'
require 'string'
require 'gnuplot'


--------------------------------------------------------------------------------
-- Set experiments configs
--------------------------------------------------------------------------------

local dataset = 'tinyshakespear'

local model = 'lstm_vanilla'

local optim_configs = {
    {expID = 'test_adam_config1', optimizer = 'adam', LR = 1e-2, LRdecay=0,   weightDecay=0},
    {expID = 'test_adam_config2', optimizer = 'adam', LR = 1e-3, LRdecay=0,   weightDecay=0},
    {expID = 'test_adam_config3', optimizer = 'adam', LR = 1e-4, LRdecay=0,   weightDecay=0},
    {expID = 'test_adam_config4', optimizer = 'adam', LR = 1e-5, LRdecay=0,   weightDecay=0},
    {expID = 'test_adam_config5', optimizer = 'adam', LR = 1e-3, LRdecay=.95, weightDecay=0},
    {expID = 'test_adam_config6', optimizer = 'adam', LR = 1e-3, LRdecay=0,   weightDecay=.95},
    {expID = 'test_adam_config7', optimizer = 'adam', LR = 1e-3, LRdecay=.95, weightDecay=.95},
}


--------------------------------------------------------------------------------
-- Train networks
--------------------------------------------------------------------------------

-- Load network trainer script
local train_net = paths.dofile('scripts/train_rnn_network.lua')

-- Load train configurations
local configs = paths.dofile('scripts/train_configs.lua')
configs.dataset = dataset
configs.model = model

for _, config in ipairs(optim_configs) do
    for k, v in pairs(config) do
        configs[k] = v
    end

    -- train network
    output[k] = train_net(configs)
end


--------------------------------------------------------------------------------
-- Combine all tests results into a plot
--------------------------------------------------------------------------------

local loss_exp = {}
for i=1, #optim_configs do
    local expID = optim_configs[i].expID

    -- load logs
    local train_loss, test_loss = {}, {}
    local fname = ('data/exp/%s/%s/epoch_loss.log'):format(dataset, expID)
    local epoch_loss = io.open(fname, 'r')
    for line in epoch_loss:lines() do
        if train_loss then
            local loss = line:split("\t")
            table.insert(train_loss, tonumber(loss[1]))
            table.insert(test_loss, tonumber(loss[2]))
        else
            -- skip the first line of the log
            train_loss = {}
            test_loss = {}
        end
    end
    epoch_loss:close()

    -- concat logs
    local optim_name = string.format('%s_LR=%2.2e_LRdecay=%0.4f_Weightdecay=%0.4f',
                                     optim_configs[i].optimizer,
                                     optim_configs[i].lr,
                                     optim_configs[i].learningRateDecay,
                                     optim_configs[i].weightDecay)

    table.insert(loss_exp, {'train_' .. optim_name, torch.range(1, #train_loss), torch.FloatTensor(train_loss), '-'})
    table.insert(loss_exp, {'test_' .. optim_name, torch.range(1, #test_loss),  torch.FloatTensor(test_loss), '+-'})
end


--------------------------------------------------------------------------------
-- Plot losses
--------------------------------------------------------------------------------

if not paths.dirp('data/tests/') then
    print('Creating dir: data/tests/')
    os.execute('mkdir -p data/tests/')
end

local fname = 'data/tests/test_adam_hyperparameters.png'
gnuplot.pngfigure('plot_labels.png')
gnuplot.title('Loss of several different optimizers for ' .. dataset)
gnuplot.plot(loss_exp)
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.grid(true)
gnuplot.plotflush()

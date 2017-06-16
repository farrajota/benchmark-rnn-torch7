--[[
    Train and evaluate a lstm network on different optimizers.
]]


require 'paths'
require 'torch'
require 'string'
require 'gnuplot'


local dataset = 'shakespear'

local optim_configs = {
    {expID = 'optimizer_test1', optimizer = 'adam', lr = 1e-2, learningRateDecay=0, weightDecay=0},
    {expID = 'optimizer_test2', optimizer = 'adam', lr = 1e-3, learningRateDecay=0, weightDecay=0},
    {expID = 'optimizer_test3', optimizer = 'adam', lr = 1e-4, learningRateDecay=0, weightDecay=0},
    {expID = 'optimizer_test4', optimizer = 'adam', lr = 1e-5, learningRateDecay=0, weightDecay=0},
    {expID = 'optimizer_test5', optimizer = 'adam', lr = 1e-3, learningRateDecay=0.95, weightDecay=0},
    {expID = 'optimizer_test6', optimizer = 'adam', lr = 1e-3, learningRateDecay=0, weightDecay=0.95},
    {expID = 'optimizer_test7', optimizer = 'adam', lr = 1e-3, learningRateDecay=0.95, weightDecay=0.95},
    {expID = 'optimizer_test8', optimizer = 'rmsprop', lr = 1e-3, learningRateDecay=0, weightDecay=0},
    {expID = 'optimizer_test9', optimizer = 'adagrad', lr = 1e-3, learningRateDecay=0, weightDecay=0},
    {expID = 'optimizer_test10', optimizer = 'adadelta', lr = 1e-3, learningRateDecay=0, weightDecay=0},
    {expID = 'optimizer_test11', optimizer = 'sgd', lr = 1e-3, momemtum = 0.9, learningRateDecay=0, weightDecay=5e-4},
}


--[[ Train networks]]
for i=1, #optim_configs do
    local str_args = ''
    for k, v in pairs(optim_configs[i]) do
        str_args = str_args .. ('-%s %s '):format(k, v)
    end

    local command = string.format('th train.lua ' .. str_args)
    print('==> Start testing optimizer %d/%d:')
    print(command)
    os.execute(command)
end


--[[ Combine all tests results into a plot ]]
local loss_exp = {}
for i=1, #optim_configs do
    local expID = optim_configs[i].expID

    -- load logs
    local fname = paths.concat('data/exp/', dataset, expID, 'epoch_loss.log')
    local epoch_loss = io.open(fname, 'r')
    local f = 0
    local train_loss, test_loss = {}, {}
    for line in epoch_loss:lines() do
        if f > 0 then
            local loss = line:split("\t")
            table.insert(train_loss, tonumber(loss[1])
            table.insert(test_loss, tonumber(loss[2])
        end
        -- increment counter
        f = f + 1
    end
    epoch_loss:close()

    -- concat logs
    local optim_name = string.format('%s_LR=%0e_LRdecay=%0.4f_Weightdecay=%0.4f',
            optim_configs[i].optimizer, optim_configs[i].lr,
            optim_configs[i].learningRateDecay, optim_configs[i].weightDecay)
        torch.FloatTensor(train_loss),
        torch.FloatTensor(test_loss)
    }
    table.insert(loss_exp, {'train_' .. optim_name, torch.FloatTensor(train_loss)})
    table.insert(loss_exp, {'test_' .. optim_name, torch.FloatTensor(test_loss)})
end


--[[ Plot losses ]]
local fname = paths.concat('data/exp/', 'test_optimizers.png')
gnuplot.pngfigure('plot_labels.png')
gnuplot.title('Loss of several different optimizers for ' .. dataset)
gnuplot.plot(loss_exp)
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.grid(true)
gnuplot.plotflush()
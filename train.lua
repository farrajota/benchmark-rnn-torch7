--[[
    Train and test a network on a dataset.
]]


require 'paths'
require 'torch'
require 'string'
require 'optim'

paths.dofile('data.lua')
paths.dofile('load_model.lua')

local tnt = require 'torchnet'

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Options
--------------------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model on Torch7.')
cmd:text()
cmd:text('Options')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-expID',  '', 'Experiment ID')
cmd:option('-dataset',  'shakespear', 'Dataset choice: shakespear | linux | wikipedia.')
cmd:option('-manualSeed',  2, 'Manually set RNG seed')
cmd:option('-GPU',         1, 'Default preferred GPU, if set to -1: no GPU')
cmd:text()
cmd:text(' ---------- Model options --------------------------------------')
cmd:text()
cmd:option('-model',   'lstm_cudnn', 'Name of the model (see load_model.lua file for more information).')
cmd:option('-rnn_size',  {256, 256}, 'size of RNN\'s internal state')
cmd:option('-bn', false, 'Use batch normalization. Only supported with fastlstm')
cmd:option('-dropout',            0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-uniform',          0.0, 'Initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:text()
cmd:text(' ---------- Hyperparameter options -----------------------------')
cmd:text()
cmd:option('-optimizer',      'adam', 'Network optimizer:  adam | rmsprop | sgd | adadelta | adagrad.')
cmd:option('-LR',               1e-3, 'Learning rate')
cmd:option('-LRdecay',           0.0, 'Learning rate decay')
cmd:option('-momentum',          0.0, 'Momentum')
cmd:option('-weightDecay',       0.0, 'Weight decay')
cmd:option('-optMethod',      'adam', 'Optimization method: rmsprop | sgd | nag | adadelta.')
cmd:option('-threshold',       0.001, 'Threshold (on validation accuracy growth) to cut off training early')
cmd:option('-epoch_reduce',       5, 'Reduce the LR at every n epochs.')
cmd:option('-LR_reduce_factor',   5, 'Reduce the learning rate by a factor.')
cmd:text()
cmd:text(' ---------- Train options --------------------------------------')
cmd:text()
cmd:option('-nEpochs',            3, 'Number of epochs for train.')
cmd:option('-seq_length',         50, 'number of timesteps to unroll for')
cmd:option('-batchSize',          64, 'number of samples per batch')
cmd:option('-grad_clip',           5, 'clip gradients at this value')
cmd:option('-train_frac',       0.95, 'fraction of data that goes into train set')
cmd:option('-val_frac',         0.05, 'fraction of data that goes into validation set')
cmd:option('-snapshot',            1, 'Save a snapshot at every N epochs.')
cmd:option('-print_every',        50, 'Print the loss at every N steps.')
cmd:option('-plot_graph',          1, 'Plots a graph of the training and test losses. (1-true | 0-false)')
cmd:text()

local opt = cmd:parse(arg or {})
opt.num_layers = #opt.rnn_size
opt.inputsize = opt.rnn_size[1]
opt.hiddensize = opt.rnn_size
opt.expID = (opt.expID ~= '' and opt.expID) or
            string.format('length=%d_%s_size=%s_nlayers=%s_dropout=%0.2f',
            opt.seq_length, opt.model, opt.rnn_size[1], opt.num_layers, opt.dropout)
opt.save = paths.concat('data/exp/', opt.dataset, opt.expID)

torch.manualSeed(opt.manualSeed)


--------------------------------------------------------------------------------
-- Data generator
--------------------------------------------------------------------------------

-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
opt.split_fractions = {opt.train_frac, opt.val_frac, test_frac}

local data = load_data(opt)

local trainIters = data.ntrain
local testIters = data.nval

local function getIterator(mode)
    -- setup data loader
    local data = data

    -- number of iterations
    local nIters = (mode=='train' and trainIters) or testIters

    return tnt.ListDataset{
        list = torch.range(1, nIters):long(),
        load = function(idx)
            local input, target = get_sample_batch(data, (mode=='train' and 1) or 2)
            return {
                input = input,
                target = target
            }
        end
    }:iterator()
end


--------------------------------------------------------------------------------
-- Model + Loss criterion
--------------------------------------------------------------------------------

local model, criterion = load_model_criterion(data.vocab_size, opt)


--------------------------------------------------------------------------------
-- Optimizer
--------------------------------------------------------------------------------

local function optimizer(name)
    if name == 'adam' then
        return optim.adam
    elseif name == 'rmsprop' then
        return optim.rmsprop
    elseif name == 'sgd' then
        return optim.sgd
    elseif name == 'adadelta' then
        return optim.adadelta
    elseif name == 'adagrad' then
        return optim.adagrad
    else
        error('Invalid optimizer: ' .. name .. '. Valid optimizers: adam | rmsprop | sgd | adadelta | adagrad.')
    end
end


local function optimState(epoch)
    if epoch % opt.epoch_reduce == 0 then
        opt.LR = opt.LR / opt.LR_reduce_factor
    end
    return {
        learningRate = opt.LR,
        learningRateDecay = opt.LRdecay,
        momentum = opt.momentum,
        dampening = 0.0,
        weightDecay = opt.weightDecay
    }
end




--------------------------------------------------------------------------------
-- Utility functions
--------------------------------------------------------------------------------

-- Gradient clipping to try to prevent the gradient from exploding.
-- ref: https://github.com/facebookresearch/torch-rnnlib/blob/master/examples/word-language-model/word_lm.lua#L216-L233
local function clipGradients(grads, norm)
    local totalnorm = 0
    for mm = 1, #grads do
        local modulenorm = grads[mm]:norm()
        totalnorm = totalnorm + modulenorm * modulenorm
    end
    totalnorm = math.sqrt(totalnorm)
    if totalnorm > norm then
        local coeff = norm / math.max(totalnorm, 1e-6)
        for mm = 1, #grads do
            grads[mm]:mul(coeff)
        end
    end
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local timers = {
   batchTimer = torch.Timer(),
   dataTimer = torch.Timer(),
   epochTimer = torch.Timer(),
}

local meters = {
    train_err = tnt.AverageValueMeter(),
    test_err = tnt.AverageValueMeter(),
}

function meters:reset()
    self.train_err:reset()
    self.test_err:reset()
end

local loggers = {
    full_test = optim.Logger(paths.concat(opt.save,'full_test.log'), opt.continue),
    full_train = optim.Logger(paths.concat(opt.save,'full_train.log'), opt.continue),
    epoch_loss = optim.Logger(paths.concat(opt.save,'epoch_loss.log'), opt.continue),
}

loggers.full_test:setNames{'Test Loss'}
loggers.full_train:setNames{'Train Loss'}
loggers.epoch_loss:setNames{'Train Loss', 'Test Loss'}

loggers.full_test.showPlot = false
loggers.full_train.showPlot = false
loggers.epoch_loss.showPlot = false


-- set up training engine:
local engine = tnt.OptimEngine()


engine.hooks.onStart = function(state)
    state.epoch = 0
end


engine.hooks.onStartEpoch = function(state)
    print('\n**********************************************')
    print(('Starting Train epoch %d/%d'):format(state.epoch+1, state.maxepoch))
    print('**********************************************')
    state.config = optimState(state.epoch+1)
    timers.epochTimer:reset()
end


-- copy sample buffer to GPU:
local inputs, targets = torch.CudaTensor(), torch.CudaTensor()
local split_targets = nn.SplitTable(1):cuda()
engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();
    inputs:resize(state.sample.input:size() ):copy(state.sample.input)
    targets:resize(state.sample.target:size() ):copy(state.sample.target)
    state.sample.input  = inputs
    state.sample.target = split_targets:forward(targets)

    timers.dataTimer:stop()
    timers.batchTimer:reset()
end


engine.hooks.onForwardCriterion = function(state)

    local iters
    if state.training then
        iters = trainIters
        meters.train_err:add(state.criterion.output)
        loggers.full_train:add{state.criterion.output}
    else
        iters = trainIters
        meters.test_err:add(state.criterion.output)
        loggers.full_test:add{state.criterion.output}
    end

    -- display train info
    if (state.t+1) % opt.print_every == 0 or (state.t+1) == iters then
        print(string.format("epoch[%d/%d][%d/%d][batch=%d][length=%d], loss = %6.8f, time/batch = %.4fs, LR=%2.2e",
            state.epoch+1, opt.nEpochs, (state.t+1), iters, opt.batchSize, opt.seq_length,
            state.criterion.output, timers.batchTimer:time().real, opt.LR))
    end
end


engine.hooks.onUpdate = function(state)
    timers.dataTimer:reset()
    timers.dataTimer:resume()
end


local _, grads = model:parameters()

engine.hooks.onBackward = function(state)
    -- clip gradients to prevent them from exploding
    clipGradients(grads, opt.grad_clip)
end


engine.hooks.onEndEpoch = function(state)
    ---------------------------------
    -- measure test loss and error:
    ---------------------------------

    print(('Train Loss: %0.5f'):format(meters.train_err:value() ))
    local tr_loss = meters.train_err:value()
    meters:reset()
    state.t = 0


    ---------------------
    -- test the network
    ---------------------

    print('\n**********************************************')
    print(('Test network (epoch = %d/%d)'):format(state.epoch, state.maxepoch))
    print('**********************************************')
    engine:test{
        network   = model,
        iterator  = getIterator('val'),
        criterion = criterion,
    }
    local ts_loss = meters.test_err:value()
    print(('Test Loss: %0.5f'):format(meters.test_err:value() ))


    loggers.epoch_loss:add{tr_loss, ts_loss}

    ---------------------------
    -- save network to disk
    ---------------------------

    if state.epoch % opt.snapshot == 0 then
        local snapshot_filename = paths.concat(opt.save, ('model_epoch=%d_loss=%0.4f.t7'):format(state.epoch, ts_loss))
        print('> Saving model snapshot to disk: ' .. snapshot_filename)
        torch.save(snapshot_filename, {state.network, data})
    end

    timers.epochTimer:reset()
    state.t = 0
end


--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------

print('==> Train network model')
engine:train{
    network   = model,
    iterator  = getIterator('train'),
    criterion = criterion,
    optimMethod = optimizer(opt.optimizer),
    config = optimState(1),
    maxepoch = opt.nEpochs
}


--------------------------------------------------------------------------------
-- Plots
--------------------------------------------------------------------------------

if opt.plot_graph > 0 then
    print('==> Plot loss graphs')
    loggers.full_test:style{'+-', '+-'}; loggers.full_test:plot()
    loggers.full_train:style{'+-', '+-'}; loggers.full_train:plot()
    loggers.epoch_loss:style{'-', '-'}; loggers.epoch_loss:plot()
end

print('==> Script Complete.')
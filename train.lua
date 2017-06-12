--[[
    Train and test a network on a dataset.
]]


require 'paths'
require 'torch'
require 'string'


local tnt = require 'torchnet'

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Options
--------------------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-dataset',  'shakespear', 'Dataset choice: shakespear | linux | wikipedia.')
cmd:option('-manualSeed',  2, 'Manually set RNG seed')
cmd:option('-GPU',         1, 'Default preferred GPU, if set to -1: no GPU')
cmd:option('-nGPU',        1, 'Number of GPUs to use by default')
cmd:option('-nThreads',    2, 'Number of data loading threads')
cmd:text()
cmd:text(' ---------- Model options --------------------------------------')
cmd:text()
cmd:option('-model',   'lstm_rnn', 'Name of the model (see load_model.lua file for more information).')
cmd:option('-rnn_size',   256, 'size of LSTM internal state')
cmd:option('-num_layers',   2, 'number of layers in the LSTM')
cmd:option('-bn', false, 'Use batch normalization. Only supported with fastlstm')
cmd:text()
cmd:text(' ---------- Train options --------------------------------------')
cmd:text()
cmd:option('-niters',                1e4,'Number of iterations')
cmd:option('-batchSize',             32, 'number of samples per batch')
cmd:option('-optimizer',          'adam','Network optimizer: adam | sgd | adagrad | rmsprop.')
cmd:option('-learning_rate',        2e-3,'learning rate')
cmd:option('-learning_rate_decay',  0.97,'learning rate decay')
cmd:option('-decay_rate',           0.95,'decay rate for rmsprop')
cmd:option('-dropout',                 0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',             50,'number of timesteps to unroll for')
cmd:option('-batch_size',             50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',             50,'number of full passes through the training data')
cmd:option('-grad_clip',               5,'clip gradients at this value')
cmd:option('-train_frac',           0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',             0.05,'fraction of data that goes into validation set')
cmd:text()

local opt = cmd:parse(arg or {})
local opt.expDir = paths.concat('data/exp/', ('%s_size=%s_nlayers=%s'):format(model))

torch.manualSeed(opt.manualSeed)


--------------------------------------------------------------------------------
-- Setup model + loss criterion
--------------------------------------------------------------------------------

local model, criterion = paths.dofile('models/load_model.lua')(opt)


--------------------------------------------------------------------------------
-- Data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        init    = function(threadid)
                    require 'torch'
                    require 'torchnet'
                    opt = lopt
                    paths.dofile('data.lua')
                    torch.manualSeed(threadid+opt.manualSeed)
                  end,
        closure = function()

            -- setup data loader
            local data_loader = select_dataset_loader(opt.dataset, mode)
            local loader = data_loader[mode]

            -- number of iterations
            local nIters = opt.trainIters

            -- setup dataset iterator
            return tnt.ListDataset{
                list = torch.range(1, nIters):long(),
                load = function(idx)
                    local input, label = getSampleBatch(loader, opt.batchSize)
                    return {
                        input = input,
                        target = label
                    }
                end
            }:batch(1, 'include-last')
        end,
    }
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local meters = {
    train_err = tnt.AverageValueMeter(),
    train_accu = tnt.AverageValueMeter(),
    test_err = tnt.AverageValueMeter(),
    test_accu = tnt.AverageValueMeter(),
}

function meters:reset()
    self.train_err:reset()
    self.train_accu:reset()
    self.test_err:reset()
    self.test_accu:reset()
end

local loggers = {
    test = Logger(paths.concat(opt.save,'test.log'), opt.continue),
    train = Logger(paths.concat(opt.save,'train.log'), opt.continue),
    full_train = Logger(paths.concat(opt.save,'full_train.log'), opt.continue),
}

loggers.test:setNames{'Test Loss', 'Test acc.'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss', 'Train accuracy'}

loggers.test.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false


-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
    if state.training then
        state.config = optimStateFn(state.epoch+1)
        if opt.epochNumber>1 then
            state.epoch = math.max(opt.epochNumber, state.epoch)
        end
    end
end


engine.hooks.onStartEpoch = function(state)
    print('\n**********************************************')
    print(('Starting Train epoch %d/%d  %s'):format(state.epoch+1, state.maxepoch,  opt.save))
    print('**********************************************')
    state.config = optimStateFn(state.epoch+1)
end


engine.hooks.onForwardCriterion = function(state)
    if state.training then
        xlua.progress((state.t+1), nBatchesTrain)

        -- compute the PCK accuracy of the networks (last) output heatmap with the ground-truth heatmap
        --local acc = accuracy(state.network.output, state.sample.target)
        --
        --meters.train_err:add(state.criterion.output)
        --meters.train_accu:add(acc)
        --loggers.full_train:add{state.criterion.output, acc}
    else
        xlua.progress(state.t, nBatchesTest)

        -- compute the PCK accuracy of the networks (last) output heatmap with the ground-truth heatmap
        --local acc = accuracy(state.network.output, state.sample.target)
        --
        --meters.test_err:add(state.criterion.output)
        --meters.test_accu:add(acc)
    end
end


-- copy sample to GPU buffer:
local inputs, targets = cast(torch.Tensor()), cast(torch.Tensor())

engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();
    inputs:resize(state.sample.input[1]:size() ):copy(state.sample.input[1])
    targets:resize(state.sample.target[1]:size() ):copy(state.sample.target[1])

    state.sample.input  = inputs
    state.sample.target = utils.ReplicateTensor2Table(targets, opt.nOutputs)
end


local test_best_accu = 0
engine.hooks.onEndEpoch = function(state)
    ---------------------------------
    -- measure test loss and error:
    ---------------------------------

    print(('Train Loss: %0.5f; Acc: %0.5f'):format(meters.train_err:value(),  meters.train_accu:value()))
    local tr_loss = meters.train_err:value()
    local tr_accuracy = meters.train_accu:value()
    loggers.train:add{tr_loss, tr_accuracy}
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
        iterator  = getIterator('test'),
        criterion = criterion,
    }
    local ts_loss = meters.test_err:value()
    local ts_accuracy = meters.test_accu:value()
    loggers.test:add{ts_loss, ts_accuracy}
    print(('Test Loss: %0.5f; Acc: %0.5f'):format(meters.test_err:value(),  meters.test_accu:value()))


    -----------------------------
    -- save the network to disk
    -----------------------------

    --storeModel(state.network.modules[1], state.config, state.epoch, opt)
    storeModel(state.network.modules[1], state.config, state.epoch, opt)

    if ts_accuracy > test_best_accu and opt.saveBest then
        storeModelBest(state.network.modules[1], opt)
        test_best_accu = ts_accuracy
    end

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
    optimMethod = optim[opt.optMethod],
    config = optimStateFn(1),
    maxepoch = nEpochs
}

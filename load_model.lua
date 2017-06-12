--[[
    Setup/load a model.
]]


require 'nn'
require 'cudnn'
require 'rnn'
require 'rnnlib'
require 'nngraph'


------------------------------------------------------------------------------------------------------------

local function setup_rnn_module(opt)
    local rnn_module

    local str = string.lower(opt.model)
    if str == 'rnn_rnn' then
    elseif str == 'rnn_rnnlib' then
    elseif str == 'rnnrelu_cudnn' then
    elseif str == 'rnntanh_cudnn' then

    elseif str == 'lstm_rnn' then
    elseif str == 'fastlstm_rnn' then
        nn.FastLSTM.usenngraph = true -- faster
        nn.FastLSTM.bn = opt.bn
        rnn = nn.FastLSTM(inputsize, hiddensize)
    elseif str == 'lstm_rnnlib' then
    elseif str == 'lstm_cudnn' then
    elseif str == 'blstm_cudnn' then

    elseif str == 'gru_rnn' then
    elseif str == 'gru_rnnlib' then
    elseif str == 'gru_cudnn' then
    else
        error('Invalid/Undefined model name: ' .. opt.model)
    end

    return rnn_module
end

------------------------------------------------------------------------------------------------------------

local function setup_model(opt)
    assert(opt, 'Missing input arg: options')

    local model = nn.Sequential()

    -- input layer (i.e. word embedding space)
    local lookup = nn.LookupTable(#trainset.ivocab, opt.inputsize)
    lookup.maxOutNorm = -1 -- prevent weird maxnormout behaviour
    model:add(lookup) -- input is seqlen x batchsize
    if opt.dropout > 0 and not opt.gru then  -- gru has a dropout option
        model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.SplitTable(1)) -- tensor to table of tensors

    -- rnn layers
    local stepmodule = nn.Sequential() -- applied at each time-step
    local inputsize = opt.inputsize
    for i, hiddensize in ipairs(opt.hiddensize) do
        local rnn = setup_rnn_module(opt)
        if opt.gru then -- Gated Recurrent Units
            rnn = nn.GRU(inputsize, hiddensize, nil, opt.dropout/2)
        elseif opt.lstm then -- Long Short Term Memory units
            nn.FastLSTM.usenngraph = true -- faster
            nn.FastLSTM.bn = opt.bn
            rnn = nn.FastLSTM(inputsize, hiddensize)
        elseif opt.mfru then -- Multi Function Recurrent Unit
            rnn = nn.MuFuRu(inputsize, hiddensize)
        else -- simple recurrent neural network
            local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
                :add(nn.ParallelTable()
                    :add(i==1 and nn.Identity() or nn.Linear(inputsize, hiddensize)) -- input layer
                    :add(nn.Linear(hiddensize, hiddensize))) -- recurrent layer
                :add(nn.CAddTable()) -- merge
                :add(nn.Sigmoid()) -- transfer
            rnn = nn.Recurrence(rm, hiddensize, 1)
        end

        stepmodule:add(rnn)

        if opt.dropout > 0 then
            stepmodule:add(nn.Dropout(opt.dropout))
        end

        inputsize = hiddensize
    end

    -- output layer
    stepmodule:add(nn.Linear(inputsize, #trainset.ivocab))
    stepmodule:add(nn.LogSoftMax())

    -- encapsulate stepmodule into a Sequencer
    model:add(nn.Sequencer(stepmodule))

    -- remember previous state between batches
    model:remember((opt.lstm or opt.gru or opt.mfru) and 'both' or 'eval')

    if not opt.silent then
        print"Language Model:"
        print(model)
    end

    if opt.uniform > 0 then
        for k,param in ipairs(model:parameters()) do
            param:uniform(-opt.uniform, opt.uniform)
        end
    end

    return model
end

------------------------------------------------------------------------------------------------------------

local function setup_criterion()
    local crit = nn.CrossEntropyCriterion()
end

------------------------------------------------------------------------------------------------------------

local load_model_criterion(opt)
    assert(opt)

    local model = setup_model(opt)
    local criterion = setup_criterion(opt)

    model:cuda()
    criterion:cuda()

    return model, criterion
end

------------------------------------------------------------------------------------------------------------

return load_model_criterion
--[[
    Setup/load a model.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'rnn'
require 'nngraph'

paths.dofile('modules/RNN.lua')
paths.dofile('modules/LSTM.lua')

------------------------------------------------------------------------------------------------------------

local function error_model()
    error('Invalid model name/type: ' .. opt.model)
end

------------------------------------------------------------------------------------------------------------

local function backend_type(opt)
    local str = string.lower(opt.model)
    if str:find('_cudnn') then
        return 'cudnn'
    elseif str:find('_rnn') then
        return 'rnn'
    elseif str:find('_rnn') then
        return 'rnnlib'
    elseif str:find('_vanilla') then
        return 'vanilla'
    else
        error_model()
    end
end

------------------------------------------------------------------------------------------------------------

--[[ Define the criterion used for optimization]]
local function setup_criterion()
    return nn.CrossEntropyCriterion()
end

------------------------------------------------------------------------------------------------------------

local function setup_model_vanilla(opt)
    assert(opt, 'Missing input arg: options')

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    local rnns = {}
    local view1 = nn.View(1, 1, -1):setNumInputDims(3)
    local view2 = nn.View(1, -1):setNumInputDims(2)
    local view3 = nn.View(opt.batchSize * opt.seq_length, -1):setNumInputDims(3)
    local lin = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)

    local model = nn.Sequential()
    model:add(lookup)
    if opt.dropout > 0 then
        model:add(nn.Dropout(opt.dropout))
    end
    local inputsize = opt.inputsize
    for i, hiddensize in ipairs(opt.hiddensize) do
        -- add module to the network

        local rnn
        local str = string.lower(opt.model)
        if str == 'rnn_vanilla' then
            rnn = nn.VanillaRNN(inputsize, hiddensize)
            rnn.remember_states = true
        elseif str == 'lstm_vanilla' then
            rnn = nn.VanillaLSTM(inputsize, hiddensize)
            rnn.remember_states = true
        else
            error_model()
        end
        model:add(rnn)

        if opt.dropout > 0 then
            model:add(nn.Dropout(opt.dropout))
        end

        table.insert(rnns, rnn)
        inputsize = hiddensize
    end
    model:add(view1)
    model:add(lin)
    model:add(view2)
    model.view1 = view1
    model.view2 = view2
    model.rnns = rnns

    -- monkey patch the forward function to reshape
    -- the view modules before doing the forward pass
    function model:forward(input)
        local N, T = input:size(1), input:size(2)
        self.view1:resetSize(N * T, -1)
        self.view2:resetSize(N, T, -1)
        return self:updateOutput(input)
    end

    function model:resetStates()
        for i, rnn in ipairs(self.rnns) do
            rnn:resetStates()
        end
    end


    local modelOut = nn.Sequential()
    modelOut:add(model)
    modelOut:add(view3)
    modelOut.view1 = view1
    modelOut.view2 = view2
    modelOut.view3 = view3

    function modelOut:forward(input)
        local N, T = input:size(1), input:size(2)
        self.view1:resetSize(N * T, -1)
        self.view2:resetSize(N, T, -1)
        self.view3:resetSize(N * T, -1)
        return self:updateOutput(input)
    end

    local criterion = setup_criterion()

    opt.nested_model = true  -- flag indicating the model is nested.
                             -- If true, save only the first model module.

    return modelOut, criterion
end

------------------------------------------------------------------------------------------------------------

local function setup_model_rnn(opt)
    assert(opt, 'Missing input arg: options')

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    lookup.maxOutNorm = -1 -- prevent weird maxnormout behaviour
    local lin = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)

    local model = nn.Sequential()
    model:add(lookup)
    if opt.dropout > 0 then
        model:add(nn.Dropout(opt.dropout))
    end
    model:add((nn.SplitTable(1))

    local stepmodule = nn.Sequential()
    local inputsize = opt.inputsize
    for i,hiddensize in ipairs(opt.hiddensize) do

        local rnn
        local str = string.lower(opt.model)
        if str == 'rnn_rnn' then
            local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
                :add(nn.ParallelTable()
                    :add(i==1 and nn.Identity() or nn.Linear(inputsize, hiddensize)) -- input layer
                    :add(nn.Linear(hiddensize, hiddensize))) -- recurrent layer
                :add(nn.CAddTable()) -- merge
                :add(nn.Sigmoid()) -- transfer
            rnn = nn.Recurrence(rm, hiddensize, 1)
        elseif str == 'lstm_rnn' then
            rnn =  nn.LSTM(inputsize, hiddensize)
        elseif str == 'fastlstm_rnn' then
            nn.FastLSTM.usenngraph = true -- faster
            nn.FastLSTM.bn = opt.bn
            rnn = nn.FastLSTM(inputsize, hiddensize)
        elseif str == 'gru_rnn' then
            rnn = nn.GRU(inputsize, hiddensize)
        else
            error_model()
        end

        stepmodule:add(rnn)

        if opt.dropout > 0 then
            stepmodule:add(nn.Dropout(opt.dropout))
        end

        inputsize = hiddensize
    end

    -- output layer
    stepmodule:add(lin)

    -- encapsulate stepmodule into a Sequencer
    model:add(nn.Sequencer(stepmodule))

    -- remember previous state between batches
    model:remember((opt.model == 'rnn_rnn' and 'eval') or 'both')

    local criterion = nn.SequencerCriterion(setup_criterion())

    return model, criterion
end

------------------------------------------------------------------------------------------------------------

local function setup_model_rnnlib(opt)
    assert(opt, 'Missing input arg: options')

    local criterion = setup_criterion()

    return model, criterion
end

------------------------------------------------------------------------------------------------------------

local function setup_model_cudnn(opt)
    assert(opt, 'Missing input arg: options')

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    local rnns = {}
    local view1 = nn.View(1, 1, -1):setNumInputDims(3)
    local view2 = nn.View(1, -1):setNumInputDims(2)
    local view3 = nn.View(opt.batchSize * opt.seq_length, -1):setNumInputDims(3)
    local lin = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)


    local model = nn.Sequential()
    model:add(lookup)
    if opt.dropout > 0 then
        model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Contiguous())
    local inputsize = opt.inputsize
    for i, hiddensize in ipairs(opt.hiddensize) do
        -- add module to the network

        local rnn
        local str = string.lower(opt.model)
        if str == 'rnnrelu_cudnn' then
            rnn = cudnn.RNNReLU(inputsize, hiddensize, 1, true, opt.dropout, true)
            rnn:resetDropoutDescriptor()
        elseif str == 'rnntanh_cudnn' then
            rnn = cudnn.RNNTanh(inputsize, hiddensize, 1, true, opt.dropout, true)
            rnn:resetDropoutDescriptor()
        elseif str == 'lstm_cudnn' then
            rnn = cudnn.LSTM(inputsize, hiddensize, 1, true, opt.dropout, true)
            rnn:resetDropoutDescriptor()
        elseif str == 'blstm_cudnn' then
            rnn = cudnn.BLSTM(inputsize, hiddensize, 1, true, opt.dropout, true)
            rnn:resetDropoutDescriptor()
        elseif str == 'gru_cudnn' then
            rnn = cudnn.GRU(inputsize, hiddensize, 1, true, opt.dropout, true)
            rnn:resetDropoutDescriptor()
        else
            error_model()
        end
        model:add(rnn)

        table.insert(rnns, rnn)
        inputsize = hiddensize
    end
    model:add(nn.Contiguous())
    model:add(view1)
    model:add(lin)
    model:add(view2)
    model.view1 = view1
    model.view2 = view2
    model.rnns = rnns

    -- monkey patch the forward function to reshape
    -- the view modules before doing the forward pass
    function model:forward(input)
        local N, T = input:size(1), input:size(2)
        self.view1:resetSize(N * T, -1)
        self.view2:resetSize(N, T, -1)
        return self:updateOutput(input)
    end

    function model:resetStates()
        for i, rnn in ipairs(self.rnns) do
            rnn:resetStates()
        end
    end


    local modelOut = nn.Sequential()
    modelOut:add(model)
    modelOut:add(view3)
    modelOut.view1 = view1
    modelOut.view2 = view2
    modelOut.view3 = view3

    function modelOut:forward(input)
        local N, T = input:size(1), input:size(2)
        self.view1:resetSize(N * T, -1)
        self.view2:resetSize(N, T, -1)
        self.view3:resetSize(N * T, -1)
        return self:updateOutput(input)
    end

    local criterion = setup_criterion()

    opt.nested_model = true  -- flag indicating the model is nested.
                             -- If true, save only the first model module.

    return modelOut, criterion
end

------------------------------------------------------------------------------------------------------------

function load_model_criterion(vocab_size, opt)
    assert(vocab_size)
    assert(opt)

    local model, criterion

    local backend_t = backend_type(opt)

    if backend_t == 'vanilla' then
        model, criterion = setup_model_vanilla(opt)
    elseif backend_t == 'rnn' then
        model, criterion = setup_model_rnn(opt)
    elseif backend_t == 'rnnlib' then
        model, criterion = setup_model_rnnlib(opt)
    elseif backend_t == 'cudnn' then
        model, criterion = setup_model_cudnn(opt)
    else
        error('Invalid backend: ' .. opt.model)
    end

    print('==> Print model to screen:')
    print(model)

    model = model:type(opt.dtype)
    criterion = criterion:type(opt.dtype)

    return model, criterion
end

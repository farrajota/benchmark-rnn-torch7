--[[
    Setup/load a model using different backends.

    List of available models+backends:
        (backend: nngraph)
        - rnn_vanilla
        - lstm_vanilla

        (backend: cudnn)
        - rnnrelu_cudnn
        - rnntanh_cudnn
        - lstm_cudnn
        - blstm_cudnn
        - gru_cudnn

        (backend: rnn (Element-Research))
        - rnn_rnn
        - lstm_rnn
        - fastlstm_rnn
        - gru_rnn

        (backend: rnnlib (facebook))
        - TODO
]]


require 'nn'
require 'rnn'
require 'nngraph'

paths.dofile('modules/RNN.lua')
paths.dofile('modules/LSTM.lua')

------------------------------------------------------------------------------------------------------------

local function error_msg_model()
    error('Invalid model name/type: ' .. opt.model)
end

------------------------------------------------------------------------------------------------------------

--[[ Define the criterion used for optimization ]]--
local function setup_criterion()
    return nn.CrossEntropyCriterion()
end


--------------------------------------------------------------------------------
-- Vanilla modules
--------------------------------------------------------------------------------

local function setup_model_vanilla(vocab_size, opt)
    assert(vocab_size)
    assert(opt, 'Missing input arg: options')

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    local rnns = {}
    local view1 = nn.View(1, 1, -1):setNumInputDims(3)
    local view2 = nn.View(1, -1):setNumInputDims(2)
    local view3 = nn.View(1, 1, -1):setNumInputDims(3)
    local lin = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)

    -- set network
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
            error_msg_model()
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


    -- Nest the model in order to be easier to train (lazy way)
    -- Note: the final view reshape reduces some headaches
    --       with torchnet + tensor reshaping of the criterion.
    local is_nested = true
    local modelOut = nn.Sequential()
    modelOut:add(model)
    modelOut:add(view3)
    modelOut.view1 = view1
    modelOut.view2 = view2
    modelOut.view3 = view3
    modelOut.rnns = rnns

    function modelOut:resetStates()
        for i, rnn in ipairs(self.rnns) do
            rnn:resetStates()
        end
    end

    -- set criterion
    local criterion = setup_criterion()

    return modelOut, criterion, is_nested
end


--------------------------------------------------------------------------------
-- RNN lib (Element-Research)
--------------------------------------------------------------------------------

local function setup_model_rnn(vocab_size, opt)
    assert(vocab_size)
    assert(opt, 'Missing input arg: options')

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    lookup.maxOutNorm = -1 -- prevent weird maxnormout behaviour
    local lin = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)

    local model = nn.Sequential()
    model:add(lookup)
    if opt.dropout > 0 then
        model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.SplitTable(1))

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
            error_msg_model()
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


--------------------------------------------------------------------------------
-- RNN lib (facebook)
--------------------------------------------------------------------------------

local function setup_model_rnnlib(vocab_size, opt)
    assert(vocab_size)
    assert(opt, 'Missing input arg: options')

    local criterion = setup_criterion()

    return model, criterion
end


--------------------------------------------------------------------------------
-- CUDNN backend
--------------------------------------------------------------------------------

local function setup_model_cudnn(vocab_size, opt)
    assert(vocab_size)
    assert(opt, 'Missing input arg: options')

    require 'cutorch'
    require 'cunn'
    require 'cudnn'

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    local rnns = {}
    local view1 = nn.View(1, 1, -1):setNumInputDims(3)
    local view2 = nn.View(1, -1):setNumInputDims(2)
    local view3 = nn.View(1, 1, -1):setNumInputDims(3)
    local lin = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)

    -- set network
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
            error_msg_model()
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


    -- Nest the model in order to be easier to train (lazy way)
    -- Note: the final view reshape reduces some headaches
    --       with torchnet + tensor reshaping of the criterion.
    local is_nested = true
    local modelOut = nn.Sequential()
    modelOut:add(model)
    modelOut:add(view3)
    modelOut.view1 = view1
    modelOut.view2 = view2
    modelOut.view3 = view3
    modelOut.rnns = rnns

    function modelOut:resetStates()
        for i, rnn in ipairs(self.rnns) do
            rnn:resetStates()
        end
    end

    -- set criterion
    local criterion = setup_criterion()

    return modelOut, criterion, is_nested
end


--------------------------------------------------------------------------------
-- Network setup
--------------------------------------------------------------------------------

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
        error_msg_model()
    end
end

------------------------------------------------------------------------------------------------------------

function load_model_criterion(vocab_size, opt)
    assert(vocab_size)
    assert(opt)

    -- select model backend
    local backend_t = backend_type(opt)
    if backend_t == 'vanilla' then
        return setup_model_vanilla(vocab_size, opt)
    elseif backend_t == 'rnn' then
        return setup_model_rnn(vocab_size, opt)
    elseif backend_t == 'rnnlib' then
        return setup_model_rnnlib(vocab_size, opt)
    elseif backend_t == 'cudnn' then
        opt.dtype = 'torch.CudaTensor'  -- force use of cuda
        return setup_model_cudnn(vocab_size, opt)
    else
        error('Invalid backend: ' .. opt.model)
    end
end

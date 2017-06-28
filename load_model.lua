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
        - rnn_rnnlib
        - lstm_rnnlib
        - gru_rnnlib
]]


require 'nn'
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

-- ref: https://github.com/jcjohnson/torch-rnn/blob/master/LanguageModel.lua
local function setup_model_vanilla(vocab_size, opt)
    assert(vocab_size)
    assert(opt, 'Missing input arg: options')

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    local rnns = {}
    local view1 = nn.View(1, 1, -1):setNumInputDims(3)  -- flattens the input tensor before feeding to the decoder
    local view2 = nn.View(1, -1):setNumInputDims(2)     -- recovers the flattened batch info from the decoder tensor
    local view3 = nn.View(1, 1, -1):setNumInputDims(3)  -- used only during train
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

    return modelOut, criterion
end


--------------------------------------------------------------------------------
-- RNN lib (Element-Research)
--------------------------------------------------------------------------------

local function setup_model_rnn(vocab_size, opt)
    assert(vocab_size)
    assert(opt, 'Missing input arg: options')

    require 'rnn'

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

    function model:resetStates()
        -- do nothing
    end

    -- set criterion
    local criterion = nn.SequencerCriterion(setup_criterion())

    return model, criterion
end


--------------------------------------------------------------------------------
-- RNN lib (facebook)
--------------------------------------------------------------------------------

-- ref: https://github.com/facebookresearch/torch-rnnlib/blob/master/examples/word-language-model/word_lm.lua
local function setup_model_rnnlib(vocab_size, opt)
    assert(vocab_size)
    assert(opt, 'Missing input arg: options')

    local rnnlib = require 'rnnlib'
    local mutils = require 'rnnlib.mutils'

    local use_cudnn = true

    -- setup rnn layers
    local rnn, cellfun, cellstr
    local str = string.lower(opt.model)
    if str == 'rnn_rnnlib' then
        cellfun = rnnlib.cell.RNNTanh
        cellstr = 'RNNTanh'
    elseif str == 'lstm_rnnlib' then
        cellfun = rnnlib.cell.LSTM
        cellstr = 'LSTM'
    elseif str == 'gru_rnnlib' then
        cellfun = rnnlib.cell.GRU
        cellstr = 'GRU'
    else
        error_msg_model()
    end

    if use_cudnn then
        rnn = rnnlib.makeCudnnRecurrent{
            cellstring = cellstr,
            inputsize  = opt.inputsize,
            hids       = opt.rnn_size,
        }
    else
        rnn = rnnlib.makeRecurrent{
            cellfn    = cellfun,
            inputsize = opt.inputsize,
            hids      = opt.rnn_size,
        }
    end
    -- Reset the hidden state.
    rnn:initializeHidden(opt.batchSize)

    local lut = nn.LookupTable(vocab_size, opt.inputsize)
    if opt.uniform then
        lut.weight:uniform(-opt.uniform, opt.uniform)
    end

    local decoder = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)
    decoder.bias:fill(0)
    if opt.uniform then
        decoder.weight:uniform(-opt.uniform, opt.uniform)
    end

    -- set network
    local model = nn.Sequential()
    model:add(mutils.batchedinmodule(rnn, lut))
    -- Select the output of the RNN.
    -- The RNN's forward gives a table of { hiddens, outputs }.
    model:add(nn.SelectTable(2))
    -- Select the output of the last layer, since the output
    -- of all layers are returned.
    model:add(nn.SelectTable(-1))
    -- Flatten the output from bptt x bsz x ntoken to bptt * bsz x ntoken.
    -- Note that the first dimension is actually a table, so there is
    -- copying involved during the flattening.
    model:add(nn.JoinTable(1))
    model:add(decoder)
    model.rnn = rnn

    -- Unroll the rnns.
    if not use_cudnn then
        for i = 1, #rnn.modules do
            rnn.modules[i]:extend(opt.seq_length)
        end
    end

    function model:resetStates()
        rnn:initializeHidden(opt.batchSize)
    end

    -- monkey-patch the forward method to only need to accept
    -- an input (avoid using the rnn.hiddenbuffer as input)
    function model:forward(input)
        return self:updateOutput{ rnn.hiddenbuffer, input }
    end

    -- same monkey-patching for the backward pass
    function model:backward(input, gradOutput, scale)
        scale = scale or 1
        self:updateGradInput({ rnn.hiddenbuffer, input }, gradOutput)
        self:accGradParameters({ rnn.hiddenbuffer, input }, gradOutput, scale)
        return self.gradInput
    end

    -- set criterion
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

    return modelOut, criterion
end


--------------------------------------------------------------------------------
-- Network setup
--------------------------------------------------------------------------------

local function backend_type(opt)
    local str = string.lower(opt.model)
    if str:find('_cudnn') then
        return 'cudnn'
    elseif str:find('_rnnlib') then
        return 'rnnlib'
    elseif str:find('_rnn') then
        return 'rnn'
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

    local model, criterion

    -- select model backend
    local backend_t = backend_type(opt)
    if backend_t == 'vanilla' then
        model, criterion = setup_model_vanilla(vocab_size, opt)
    elseif backend_t == 'rnn' then
        model, criterion = setup_model_rnn(vocab_size, opt)
    elseif backend_t == 'rnnlib' then
        model, criterion = setup_model_rnnlib(vocab_size, opt)
    elseif backend_t == 'cudnn' then
        opt.dtype = 'torch.CudaTensor'  -- force use of cuda
        model, criterion = setup_model_cudnn(vocab_size, opt)
    else
        error('Invalid backend: ' .. opt.model)
    end

    return model, criterion, backend_t
end

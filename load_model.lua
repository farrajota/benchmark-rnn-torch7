--[[
    Setup/load a model.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'rnn'
require 'nngraph'


------------------------------------------------------------------------------------------------------------

local function is_backend_cudnn(opt)
    local str = string.lower(opt.model)
    if str == 'rnnrelu_cudnn' then
        return true
    elseif str == 'rnntanh_cudnn' then
        return true
    elseif str == 'lstm_cudnn' then
        return true
    elseif str == 'blstm_cudnn' then
        return true
    elseif str == 'gru_cudnn' then
        return true
    else
        return false
    end
end

------------------------------------------------------------------------------------------------------------

local function rnn_module(inputsize, hiddensize, opt)
    assert(inputsize)
    assert(hiddensize)
    assert(opt)

    local str = string.lower(opt.model)
    -- rnn
    if str == 'rnn_rnn' then
        local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
            :add(nn.ParallelTable()
                :add(i==1 and nn.Identity() or nn.Linear(inputsize, hiddensize)) -- input layer
                :add(nn.Linear(hiddensize, hiddensize))) -- recurrent layer
            :add(nn.CAddTable()) -- merge
            :add(nn.Sigmoid()) -- transfer
        return nn.Recurrence(rm, hiddensize, 1)
    elseif str == 'lstm_rnn' then
        return nn.LSTM(inputsize, hiddensize)
    elseif str == 'fastlstm_rnn' then
        nn.FastLSTM.usenngraph = true -- faster
        nn.FastLSTM.bn = opt.bn
        return nn.FastLSTM(inputsize, hiddensize)
    elseif str == 'gru_rnn' then
        return nn.GRU(inputsize, hiddensize)

    -- cudnn
    elseif str == 'rnnrelu_cudnn' then
        return cudnn.RNNReLU(inputsize, hiddensize, 1, true)
    elseif str == 'rnntanh_cudnn' then
        return cudnn.RNNTanh(inputsize, hiddensize, 1, true)
    elseif str == 'lstm_cudnn' then
        return cudnn.LSTM(inputsize, hiddensize, 1, true)
    elseif str == 'blstm_cudnn' then
        return cudnn.BLSTM(inputsize, hiddensize, 1, true)
    elseif str == 'gru_cudnn' then
        return cudnn.GRU(inputsize, hiddensize, 1, true)
    else
        error('Invalid/Undefined model name: ' .. opt.model)
    end
end

------------------------------------------------------------------------------------------------------------

local function setup_rnn_module(opt)
    local rnn_layers = nn.Sequential()
    local inputsize = opt.inputsize
    for i, hiddensize in ipairs(opt.hiddensize) do
        -- add module to the network
        rnn_layers:add(rnn_module(inputsize, hiddensize, opt))

        -- add dropout (if enabled)
        if opt.dropout > 0 then
            rnn_layers:add(nn.Dropout(opt.dropout))
        end

        inputsize = hiddensize
    end
    return rnn_layers
end

------------------------------------------------------------------------------------------------------------

local function setup_model(vocab_size, opt)
    assert(opt, 'Missing input arg: options')

    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    local rnn = setup_rnn_module(opt)
    local view1 = nn.View(1, 1, -1):setNumInputDims(3)
    local view2 = nn.View(1, -1):setNumInputDims(2)
    local view3 = nn.View(1, -1):setNumInputDims(3)
    local lin = nn.Linear(opt.hiddensize[opt.num_layers], vocab_size)

    local model = nn.Sequential()
    model:add(lookup)
    model:add(nn.Contiguous())
    model:add(rnn)
    model:add(nn.Contiguous())
    model:add(view1)
    model:add(lin)
    model:add(view2)

    -- monkey patch the forward function to reshape
    -- the view modules before doing the forward pass
    lookup.updateOutput_new = function(module, input)
        local N, T = input:size(1), input:size(2)
        model.view1:resetSize(N * T, -1)
        model.view2:resetSize(N, T, -1)
        return lookup:forward(input)
    end

    if opt.uniform > 0 then
        for k,param in ipairs(model:parameters()) do
            param:uniform(-opt.uniform, opt.uniform)
        end
    end

    print('==> Print model to screen:')
    print(model)

    local modelOut = nn.Sequential()
    modelOut:add(model)
    modelOut:add(view3)
    modelOut.view1 = view1
    modelOut.view2 = view2
    modelOut.view3 = view3

    return modelOut
end

------------------------------------------------------------------------------------------------------------

local function setup_criterion()
    --local crit = nn.ClassNLLCriterion()
    --local criterion = nn.SequencerCriterion(crit)
    local criterion = nn.CrossEntropyCriterion()
    return criterion
end

------------------------------------------------------------------------------------------------------------

function load_model_criterion(vocab_size, opt)
    assert(vocab_size)
    assert(opt)

    local model = setup_model(vocab_size, opt)
    local criterion = setup_criterion()

    model = model:type(opt.dtype)
    criterion = criterion:type(opt.dtype)

    return model, criterion
end

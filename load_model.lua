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
        return cudnn.RNNReLU(inputsize, hiddensize, 1)
    elseif str == 'rnntanh_cudnn' then
        return cudnn.RNNTanh(inputsize, hiddensize, 1)
    elseif str == 'lstm_cudnn' then
        return cudnn.LSTM(inputsize, hiddensize, 1)
    elseif str == 'blstm_cudnn' then
        return cudnn.BLSTM(inputsize, hiddensize, 1)
    elseif str == 'gru_cudnn' then
        return cudnn.GRU(inputsize, hiddensize, 1)
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

    -- input layer (i.e. word embedding space)
    local lookup = nn.LookupTable(vocab_size, opt.inputsize)
    lookup.maxOutNorm = -1 -- prevent weird maxnormout behaviour

    -- rnn layers
    local rnn_layers = setup_rnn_module(opt)

    -- output layer
    local classifier = nn.Sequential()
    classifier:add(nn.Linear(opt.hiddensize[opt.num_layers], vocab_size))

    local model = nn.Sequential()
    -- add input layer
    model:add(lookup)
    if opt.dropout > 0 then
        model:add(nn.Dropout(opt.dropout))
    end
    -- add rnn layers
    if is_backend_cudnn(opt) then
        model:add(rnn_layers)
        model:add(nn.SplitTable(1))
    else
        model:add(nn.SplitTable(1)) -- tensor to table of tensors
        -- encapsulate rnn layers into a Sequencer
        model:add(nn.Sequencer(rnn_layers))
    end
    -- encapsulate classifier into a Sequencer
    model:add(nn.Sequencer(classifier))
    if not is_backend_cudnn(opt) then
      -- remember previous state between batches
      model:remember()
    end

    if opt.uniform > 0 then
        for k,param in ipairs(model:parameters()) do
            param:uniform(-opt.uniform, opt.uniform)
        end
    end

    print('==> Print model to screen:')
    print(model)

    return model
end

------------------------------------------------------------------------------------------------------------

local function setup_criterion()
    local crit = nn.CrossEntropyCriterion()
    local criterion = nn.SequencerCriterion(crit)
    return criterion
end

------------------------------------------------------------------------------------------------------------

function load_model_criterion(vocab_size, opt)
    assert(vocab_size)
    assert(opt)

    local model = setup_model(vocab_size, opt)
    local criterion = setup_criterion()

    model:cuda()
    criterion:cuda()

    return model, criterion
end

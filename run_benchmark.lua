--[[
    Train and evaluate a series of different rnn modules on three datasets (Shakespear, Linux kernel and Wikipedia).
]]


require 'paths'
require 'torch'


local datasets = {'shakespear', 'linux', 'wikipedia'}

local models = {
    'rnn_rnn', 'rnn_rnnlib', 'rnnrelu_cudnn', 'rnntanh_cudnn',
    'lstm_rnn', 'fastlstm_rnn', 'lstm_rnnlib', 'lstm_cudnn', 'blstm_cudnn'
    'gru_rnn', 'gru_rnnlib', 'gru_cudnn',
}

--------------------------------------------------------------------------------
-- Benchmark networks
--------------------------------------------------------------------------------

for k, set in pairs(datasets) do
    print('==> Start models\' train/eval on the dataset: ' .. set)

    for _, model in pairs(models) do
        -- train/test network
        print('> Train + eval model: ' .. model)
        os.execute(('th train.lua -model %s'):format(model))
    end

    -- process plots
    -- save plots to disk
end


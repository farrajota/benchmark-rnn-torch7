local models = {}

--[[ Vanilla nngraph models ]]
models['vanilla_rnn'] = function() end
models['vanilla_lstm'] = function() end

--[[ rnn library models ]]
models['rnn_rnn'] = function() end
models['rnn_lstm'] = function() end
models['rnn_fastlstm'] = function() end
models['rnn_gru'] = function() end

--[[ cudnn library models ]]
models['cudnn_rnn_relu'] = function() end
models['cudnn_rnn_tanh'] = function() end
models['cudnn_lstm'] = function() end
models['cudnn_blstm'] = function() end
models['cudnn_gru'] = function() end

--[[ torch-rnnlib library models]]
-- TODO
models['rnnlib_rnn'] = function() end
models['rnnlib_lstm'] = function() end
models['rnnlib_gru'] = function() end

return models
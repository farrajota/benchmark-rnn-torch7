--[[
    Global configurations used for training all networks on the
    different datasets.
]]


if not g_hidden_dimension_size then
    g_hidden_dimension_size = 256  -- default hidden dimension size
end

if not g_dataset then
    g_dataset = 'shakespear'  -- default dataset
end


local configs = {
    -- General options
    expID = '',
    dataset = g_dataset,
    manualSeed = 2,
    GPU = 1,  -- use GPU

    -- Model options
    model = 'rnn_vanilla',
    rnn_size = g_hidden_dimension_size,
    num_layers = 2,
    epoch_reduce = 0,
    LR_reduce_factor = 0,

    -- Hyperparameter options
    optimizer = 'adam',
    LR = 1e-3,
    epoch_reduce = 5,
    LR_reduce_factor = 5,

    -- Train options
    nEpochs = 10,
    seq_length = 50,
    batchSize = 64,
    grad_clip = 5,
    train_frac = 0.95,
    val_frac = 0.05,
    snapshot = 5,
    print_every = 50,
    plot_graph = 0
}

return configs
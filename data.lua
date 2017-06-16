--[[
    Preprocess, load and fetch data samples.
]]


require 'lfs'


------------------------------------------------------------------------------------------------------------

local function is_dataset_valid(dataset)
    assert(dataset, 'Missing argument: dataset')

    if dataset == 'shakespear' then
        return true
    elseif dataset == 'linux' then
        return true
    elseif dataset == 'wikipedia' then
        return true
    else
        error('Invalid dataset: ' .. dataset ..'. Available datasets: shakepsear | linux | wikipedia.')
    end
end

------------------------------------------------------------------------------------------------------------

local function convert_txt_to_tensor(in_textfile, out_vocabfile, out_tensorfile)
    assert(in_textfile)
    assert(out_vocabfile)
    assert(out_tensorfile)

    local timer = torch.Timer()

    print('loading text file...')
    local cache_len = 10000
    local rawdata
    local tot_len = 0
    local f = assert(io.open(in_textfile, "r"))

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')

    -- record all characters to a set
    local unordered = {}
    rawdata = f:read(cache_len)
    repeat
        for char in rawdata:gmatch'.' do
            if not unordered[char] then unordered[char] = true end
        end
        tot_len = tot_len + #rawdata
        rawdata = f:read(cache_len)
    until not rawdata
    f:close()

    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered)

    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end

    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
    f = assert(io.open(in_textfile, "r"))
    local currlen = 0
    rawdata = f:read(cache_len)
    repeat
        for i=1, #rawdata do
            data[currlen+i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
        end
        currlen = currlen + #rawdata
        rawdata = f:read(cache_len)
    until not rawdata
    f:close()

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)
end

------------------------------------------------------------------------------------------------------------

function load_data(opt)
    assert(opt)
    assert(is_dataset_valid(opt.dataset))

    local input_file = paths.concat('data/', opt.dataset .. '_input.txt')
    local vocab_file = paths.concat('data/', opt.dataset .. '_vocab.t7')
    local tensor_file = paths.concat('data/', opt.dataset .. '_data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end

    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        convert_txt_to_tensor(input_file, vocab_file, tensor_file)
    end


    print('loading data files...')
    local data = torch.load(tensor_file)
    local out = {}
    out.vocab_mapping = torch.load(vocab_file)

    -- cut off the end so that it divides evenly
    local batch_size = opt.batchSize
    local seq_length = opt.seq_length
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    out.vocab_size = 0
    for _ in pairs(out.vocab_mapping) do
        out.vocab_size = out.vocab_size + 1
    end

    -- out.batches is a table of tensors
    print('reshaping tensor...')
    out.batch_size = batch_size
    out.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    out.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    --out.x_batches =  data:view(batch_size, -1):transpose(1,2):split(seq_length, 1)
    out.nbatches = #out.x_batches
    out.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    --out.y_batches = ydata:view(batch_size, -1):transpose(1,2):split(seq_length, 1)
    assert(#out.x_batches == #out.y_batches)

    -- lets try to be helpful here
    if out.nbatches < opt.batchSize then
        print('WARNING: less than ' .. opt.batchSize .. ' batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    local split_fractions = opt.split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')

    if split_fractions[3] == 0 then
        -- catch a common special case where the user might not want a test set
        out.ntrain = math.floor(out.nbatches * split_fractions[1])
        out.nval = out.nbatches - out.ntrain
        out.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        out.ntrain = math.floor(out.nbatches * split_fractions[1])
        out.nval = math.floor(out.nbatches * split_fractions[2])
        out.ntest = out.nbatches - out.nval - out.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    out.split_sizes = {out.ntrain, out.nval, out.ntest}
    out.batch_ix = {0,0,0}

    print(string.format('> Data load done. Number of data batches in train: %d, val: %d, test: %d',
         out.ntrain, out.nval, out.ntest))

    collectgarbage()

    return out
end

------------------------------------------------------------------------------------------------------------

function get_sample_batch(data, split_index)
    assert(data)
    assert(split_index)

    if data.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        error('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
    end

    -- split_index is integer: 1 = train, 2 = val, 3 = test
    data.batch_ix[split_index] = data.batch_ix[split_index] + 1
    if data.batch_ix[split_index] > data.split_sizes[split_index] then
        data.batch_ix[split_index] = 1 -- cycle around to beginning
    end

    -- pull out the correct next batch
    local ix = data.batch_ix[split_index]
    if split_index == 2 then ix = ix + data.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + data.ntrain + data.nval end -- offset by train + val
    return data.x_batches[ix], data.y_batches[ix]
end

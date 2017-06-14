--[[
    Sample characters from a trained model.
]]

require 'paths'
require 'torch'
require 'string'

paths.dofile('load_model.lua')

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Options
--------------------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model.')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model', 'model checkpoint to use for sampling')
-- optional parameters
cmd:option('-manualSeed',   2,'random number generator\'s seed')
cmd:option('-sample',       1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',   "",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',    2000,'number of characters to sample')
cmd:option('-temperature',  1,'temperature of sampling')
cmd:text()

local opt = cmd:parse(arg or {})

torch.manualSeed(opt.manualSeed)


--------------------------------------------------------------------------------
-- Model
--------------------------------------------------------------------------------

local checkpoint = torch.load(opt.model)
local model = checkpoint[1]
model:evaluate()


--------------------------------------------------------------------------------
-- Initialize the vocabulary (and its inverted version)
--------------------------------------------------------------------------------

local data = checkpoint[2]
local vocab = data.vocab_mapping
local ivocab = {}
for c, i in pairs(vocab) do ivocab[i] = c end


--------------------------------------------------------------------------------
-- Sample characters
--------------------------------------------------------------------------------

-- get first character to start predicting other characters
if string.len(opt.primetext) > 0 then
    print('seeding with ' .. seed_text)
    print('--------------------------')
    for c in seed_text:gmatch'.' do
        prev_char = torch.Tensor{vocab[c]}
        io.write(ivocab[prev_char[1]])
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do
            table.insert(current_state, lst[i])
        end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
else
    -- fill with uniform probabilities over characters (? hmm)
    print('Seed text empty or missing, using uniform probability over first character.')
    print('--------------------------')
    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
end

-- start sampling/argmaxing
for i=1, opt.length do

    -- log probabilities from the previous timestep
    if opt.sample == 0 then
        -- use argmax
        local _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        prediction:div(opt.temperature) -- scale by temperature
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    end

    -- forward the rnn for next character
    local lst = model:forward(prev_char:view(1,1))
    prediction = lst[#lst] -- last element holds the log probabilities

    io.write(ivocab[prev_char[1]])
end
io.write('\n')
io.flush()

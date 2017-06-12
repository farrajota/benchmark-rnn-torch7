--[[
    This script will download and setup the necessary data to disk.
]]


require 'paths'
require 'torch'

local savepath = './data/'

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end


--------------------------------------------------------------------------------
-- Download data
--------------------------------------------------------------------------------

-- shakespear
local shakespear_url = 'http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt'
local shakespear_filename = paths.concat(savepath, 'shakespear_input.txt')
print('Download the Shakespear text file to disk: ', shakespear_filename)
os.execute(('wget -O %s %s'):format(shakespear_filename, shakespear_url))

-- linux kernel
local linux_url = 'http://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt'
local linux_filename = paths.concat(savepath, 'linux_input.txt')
print('Download the concatenated Linux kernel text file to disk: ', linux_filename)
os.execute(('wget -O %s %s'):format(linux_filename, linux_url))

-- wikipedia
local wikipedia_url = 'http://mattmahoney.net/dc/enwik8.zip'
local wikipedia_zip = paths.concat(savepath, 'enwik8.zip')
local wikipedia_filename = paths.concat(savepath, 'wikipedia_input.txt')
print('Download the Wikipedia dataset to disk: ', wikipedia_filename)
os.execute(('wget -O %s %s'):format(wikipedia_zip, wikipedia_url))

-- unzip
os.execute(('unzip %s'):format(wikipedia_zip))
os.execute(('mv enwik8 %s'):format(wikipedia_filename))
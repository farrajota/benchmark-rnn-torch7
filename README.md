# RNN modules/libraries benchmark on Torch7

Train and test rnn modules like (rnn, lstm, gru, etc.) of different libraries on a simple task for word language model.

The evaluated rnn modules on this repo are the following:

- RNN
- LSTM
- BLSTM
- GRU

## Requirements

To use this repository you must have [Torch7](http://torch.ch/) installed on your system. It is best to have an up-to-date version of torch7 before running this code. To do this, simply go to your `torch/` folder and run the `./update.sh` file.

Also, you'll need a NVIDIA GPU with compute capability 3.5+ (2GB+ ram) and `CUDNN R5` installed in order to run this code.

Next, you'll need to install the following dependencies for torch7:

```bash
luarocks install cutorch
luarocks install cudnn
luarocks install torchnet
luarocks install tds
```


### Torch7 RNN libraries

This repository makes use of the following libraries for evaluation of several implementations of rnn modules:

- [rnn](https://github.com/Element-Research/rnn)
- [cudnn](https://github.com/soumith/cudnn.torch)


To install these packages just run the following command:

```bash
luarocks install rnn
luarocks install cudnn
```

## Getting started

### Data setup

Download and setup the necessary data by running the following script:

```
th download_setup_data.lua
```

This script will download and extract the following datasets to disk:

- [shakespear](http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt)
- [linux kernel](http://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt)
- [wikipedia](http://prize.hutter1.net/)


### Training and testing the networks

To evaluate the different rnn modules tested here, run the main script:

```
th run_benchmark.lua
```

This will train and test several networks on three different datasets and plot all results into a graph for each dataset.


### Results

When running the benchmark script, you should get the same results presented in these next graphs.

#### Shakespear

#### linux kernel

#### wikipedia

## License

MIT license (see the LICENSE file)

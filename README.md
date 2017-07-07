# RNN modules/libraries benchmark on Torch7

This repo contains benchmark results for popular rnn modules/architectures on three different libraries available for Torch7 on a simple task for word language model:

- [cudnn](https://github.com/soumith/cudnn.torch)
- [rnn](https://github.com/Element-Research/rnn)
- [rnnlib](https://github.com/facebookresearch/torch-rnnlib)

The evaluated rnn architectures are the following:

- RNN
- LSTM
- GRU

## Requirements

To use this repository you must have [Torch7](http://torch.ch/) installed on your system.
Also, you'll need a NVIDIA GPU with compute capability 3.5+ (2GB+ ram) and `CUDNN R5+` installed in order to run this code.

Furthermore, you'll need to install the following dependencies for torch7:

```bash
luarocks install cudnn
luarocks install optnet
luarocks install rnn
luarocks install rnnlib
luarocks install torchnet
```

> Note: Please make sure you have an up-to-date version of torch7 before running the benchmark script. To do this, go to your `torch/` folder and run the `./update.sh` file.


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


### Benchmarking the networks/libraries

To evaluate the different rnn modules/libraries tested here, run the main script:

```
th scripts/run_benchmark.lua
```

This will train and test several networks on three different datasets and plot all results into sveral graphs for each dataset.


### Results

When running the benchmark script, you should get the same results presented in these next graphs.

All models have a sequence length of 50, a batch size of 64, 2 layers, and were averaged over 10 epochs and over all 3 datasets.

> Warning: Results are currently being processed. Once they are finished (it takes several days!) I'll upload them.

#### Speed (batch/ms)

![batch](data/results/speed_vs_dimension.png "batch vs dim")

#### Forward/backward Speed (batch/ms)

![fw_bw](data/results/fw_bw_vs_dimension.png "forward/backward vs dim")


#### GPU Memory Usage (in MB)

![gpu_memory](data/results/memory_vs_dimension.png "GPU memory vs dim")


#### Test Loss Results

Below are the results of the loss of all rnn
models tested on the Shakespear,
 Linux kernel and Wikipedia datasets.

| Shakespear | Linux kernel | Wikipedia |
| --- | --- | --- |
| ![256 shakespear](data/results/loss_shakespear_256.png "loss w/ hidden dimension 256") | ![256 linux](data/results/loss_linux_256.png "loss w/ hidden dimension 256") | ![256 wikipedia](data/results/loss_wikipedia_256.png "loss w/ hidden dimension 256") |
| ![512 shakespear](data/results/loss_shakespear_512.png "loss w/ hidden dimension 512") | ![512 linux](data/results/loss_linux_512.png "loss w/ hidden dimension 512") | ![512 wikipedia](data/results/loss_wikipedia_512.png "loss w/ hidden dimension 512") |
| ![1024 shakespear](data/results/loss_shakespear_1024.png "loss w/ hidden dimension 1024") | ![1024 linux](data/results/loss_linux_1024.png "loss w/ hidden dimension 1024") | ![1024 wikipedia](data/results/loss_wikipedia_1024.png "loss w/ hidden dimension 1024") |
| ![2048 shakespear](data/results/loss_shakespear_2048.png "loss w/ hidden dimension 2048") | ![2048 linux](data/results/loss_linux_2048.png "loss w/ hidden dimension 2048") | ![2048 wikipedia](data/results/loss_wikipedia_2048.png "loss w/ hidden dimension 2048") |
| ![4096 shakespear](data/results/loss_shakespear_4096.png "loss w/ hidden dimension 4096") | ![4096 linux](data/results/loss_linux_4096.png "loss w/ hidden dimension 4096") | ![4096 wikipedia](data/results/loss_wikipedia_4096.png "loss w/ hidden dimension 4096") |


## License

MIT license (see the LICENSE file)

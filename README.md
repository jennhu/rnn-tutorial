# An Introduction to Recurrent Neural Networks

Recurrent neural networks (RNNs) have become increasingly popular in machine learning for handling sequential data. In this tutorial, we will cover background about the architecture, a toy training example, and a demo for evaluating a larger pre-trained model.

These materials are prepared for the [Harvard-MIT Theoretical and Computational Neuroscience Journal Club](https://compneurojc.github.io/) and should be beginner-friendly!

The RNN multitask training repo is imported from https://github.com/benhuh/RNN_multitask.


**Date**: June 26, 2019

**Authors**: Jennifer Hu (MIT), Ben Huh (IBM), Peng Qian (MIT)

---

## Prerequisites

**NOTE: If you are cloning this repository, make sure to use the flag `--recurse-submodules` or `--recursive` depending on your version of Git (see https://stackoverflow.com/a/4438292).**

To run this tutorial, you will need to install [Pytorch](https://pytorch.org/), [numpy](https://www.numpy.org/), and [Jupyter](https://jupyter.org/). See the notebook `demo.ipynb` for more details.

To evaluate the pre-trained model, you will need to download the checkpoint from [this link](https://www.dropbox.com/s/er9exdbwun4rex9/model_bnc.pt?dl=1).
Alternatively, you can run
```bash
wget https://www.dropbox.com/s/er9exdbwun4rex9/model_bnc.pt?dl=1
```

Make sure to move the downloaded file to the `materials` folder.

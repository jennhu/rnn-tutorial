# An Introduction to Recurrent Neural Networks

Recurrent neural networks (RNNs) have become increasingly popular in machine learning for handling sequential data. In this tutorial, we will cover background about the architecture, a toy training example, and a demo for evaluating a larger pre-trained model.

These materials were prepared for the Harvard-MIT Theoretical and Computational Neuroscience Journal Club and should be beginner-friendly!

**Date**: June 26, 2019

**Authors**: Jennifer Hu (MIT) and Ben Huh (IBM)

---

## Prerequisites

To run this tutorial, you will need to install [Pytorch](https://pytorch.org/) and [numpy](https://www.numpy.org/). See the notebook `demo.ipynb` for more details.

To evaluate the pre-trained model, you will need to download the checkpoint from [this link](https://www.dropbox.com/s/er9exdbwun4rex9/model_bnc.pt?dl=1).
Alternatively, you can run
```bash
wget https://www.dropbox.com/s/er9exdbwun4rex9/model_bnc.pt?dl=1
```

Make sure to move the downloaded file to the `materials` folder.

### DCGAN - Deep Convolutional Generative Adversarial Networks
> A implementation from scratch

&nbsp;

#### Reading Notes
- Generator
  - Uses fractional-strided convolutions (e.g: transposed convolutions) for upsampling
  - Uses ReLU (except for Tanh at output layer)
  - Uses BatchNorm (except for output layer)
  - Not using fully connected or pooling layers.
- Discriminator:
  - Uses strided convolutions for downsampling
  - Uses Leaky ReLU (except for Sigmoid at output layer)
  - Uses Batch Normalization (except for input layer)
  - Not using fully connected or pooling layers.

#### References
- [Article](https://arxiv.org/pdf/1511.06434.pdf)
- [IMG/Architecture](https://debuggercafe.com/wp-content/uploads/2020/07/dcgan_overall.png)
# SANet
This repository contains the code (in [TensorFlow](https://www.tensorflow.org/)) for the paper:

[**Arbitrary Style Transfer with Style-Attentional Networks**](https://arxiv.org/abs/1812.02342)
<br>
Dae Young Park,
Kwang Hee Lee
<br>

Arbitrary style transfer aims to synthesize a content image with the style of an image to create a third image that has never been seen before. Recent arbitrary style transfer algorithms find it challenging to balance the content structure and the style patterns. Moreover, simultaneously maintaining the global and local style patterns is difficult due to the patch-based mechanism. In this paper, authors introduce a novel style-attentional network (SANet) that efficiently and flexibly integrates the local style patterns according to the semantic spatial distribution of the content image. A new identity loss function and multi-level feature embeddings enable SANet and decoder to preserve the content structure as much as possible while enriching the style patterns. Experimental results demonstrate that algorithm synthesizes stylized images in real-time that are higher in quality than those produced by the state-of-the-art algorithms.

<p align='center'>
  <img src='_assets/architecture.png' width="600px">
</p>


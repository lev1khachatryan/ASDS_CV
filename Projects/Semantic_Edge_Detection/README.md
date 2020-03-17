# Semantic Edge Detection.

* Choose a video from the DAVIS dataset, get frames, obtain the edges of the frames from the annotations of these frames.

* Build an Encoder-Decoder architecture to train a neural network for semantic edge detection on a single image. Take as encoder one of the ResNet (18, 50 or 101) architectures and initialize it with a pre-trained model.

* Define a loss function, overfit the model with on the chosen video’s frames.

* As can be noticed, in the ResNet architecture there is a MaxPool(x) layer. Define a function F(x) as a mini-network (with weights W ), such that for some W* the function F(x) is identical with the function MaxPool(x) . The mini-network F(x) must be composed of only convolutional layers and pointwise non-linearities. Replace the MaxPool(x) function with the F(x) and initialize it with W*.

* Build an Encoder-Decoder architecture to train a neural network which will take into account the result of itself on the previous frame of the chosen video.

* Take as encoder of the above mentioned neural network NN the same architecture as in step 2. (except for the first convolution layer, which will accept 4-channeled input instead of 3-channeled). Initialize the encoder with the pre-trained resnet weights except for the kernel of the first convolution. For the kernel of the first convolution make the following initialization: the kernel is a tensor with the shape (H, W , 4, OutChannels) , initialize it’s slice [:, :, : 3, :] with the pretrained resnet’s first convolution kernel, and the slice [:, :, 3, :] randomly.

* Train the NN() network on the k − length frame sequences from the chosen video (try to overfit on the chosen video).

* Add perceptual losses to the training.

* Add non-maximum suppression after the output of the network. The non-maximum suppression must be a part of the main graph and not just a post-processing function (to be able to put loss after the non-maximum suppression operation and train the network with this op.).


## My Running Environment

### Hardware

* CPU: Intel® Core™ i5-8250U (1.60GHz x 8 cores, 16 threads)

* GPU: NVIDIA® GeForce GTX 1080/PCle/SSE2

* Memory: 8GB GiB

* OS type: 64-bit

* Disk: 1.2 TB


### Operating System

Windows 10

# Input Data

You can find all necessary videos in the link below:
https://drive.google.com/open?id=1E0oyBoyzWx28cOvvmB_NbogLIbXPBsUf

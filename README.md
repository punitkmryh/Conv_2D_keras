# A Beginner’s Guide to Convolution NeuNet (CNN)

![CNN](/Users/nitishharsoor/Prop/CNN-trans/cnn.jpeg)

## 1. Introduction

A **Convolutional Neural Network (ConvNet/CNN)** is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

* The pre-processing required in a ConvNet is much lower as compared to other classification algorithms.

> **`ROLE:`**
>
> *The role of the ConvNet is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction.*

* In Conv-Net, multiple filters are taken to slice through the image and map them one by one and learn different portions of an input image.
* Imagine a small filter sliding left to right across the image from top to bottom and that moving filter is looking for, say, a dark edge. Each time a match is found, it is mapped out onto an output image.
* Note that an image is 2 dimensional with width and height.
  *  If the image is colored, it is considered to have one more dimension for RGB color, hence **3 channels for each color**,
  *  For that reason, `2D convolutions` are usually used for black and white images, 
  *  while `3D convolutions` are used for colored images. 
* A filter is a `3 x 3` matrix of weights that is slid over an image in order to produce a filtered output.



### <u>1.1. Overview of Convo-Neu-Net</u>

![handwritting-cnn](/Users/nitishharsoor/Prop/CNN-trans/cnn-handwrite.jpeg)

In the above figure, ConNet learns image features step by step, 

1. Firstly input image is taken to each neurons.
2. Then in **`Conv_1`**  Convolution layer,  **Filter | Kernel | Patch** is applied to overall input image by learning edges, bumps, lines at angle form the every pixels of input image which is stored as **Feature map**.
3. Input image is then reduced to lower resolution using **`max-pooling`**layer.
4. Now additional Convolutional layer **`Conv_2`** is add to learn from smaller resolution image **`n1 channel`** by looking into connection between lines and bumps from previous **Feature map**.
5. At the end after **`max-pooling`** layer, where image is **flattened** and given to **`fully-connected | Dense`** layer with ReLU activation function.
6. On passing softmax last layer it ends with ouput layer contains the label which is in the form of one-hot encoded. in above figure it gives 0 to 9 labels.

------

## 2. Few Definitions:

Below are few definition before we move on to understand Conv_Neu_Net:

###  <u>Filters | Kernel | Patch</u> 

* The filters are the **“`neurons`”** of the layer.
* A filter is a square-shaped object that scans over the image.
* If the convolutional layer is an **input layer,** then the input patch will be `pixel values`.
* if deeper layer in the architecture, then the convolutional layer will take input from a feature map from the previous layer.

	>  **Image Dimensions** = 5 (**Height**) x 5 (**Breadth**) x 1 (Number of **channels**, eg. RGB)

![kernel](/Users/nitishharsoor/Prop/CNN-trans/cnn-filter.gif) 							<img src="/Users/nitishharsoor/Prop/CNN-trans/kernel-moment.png" alt="kernel-movment" style="zoom:50%;" />

[^a) CNN Kernek/filter, b) Kernel Moment]: In the above demonstration, the green section resembles our **5x5x1 input image, I**. The element involved in carrying out the convolution operation in the first part of a Convolutional Layer is called the **Kernel/Filter, K**, represented in the color yellow. We have selected **K as a 3x3x1 matrix.
Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters. The below example shows various convolution image after applying different types of filters (Kernels).

![Image for post](/Users/nitishharsoor/Prop/CNN-trans/common-filter.png)

### <u>Feature Maps</u>

- A given filter is drawn across the entire previous layer, moved one pixel at a time. Each position results in an activation of the neuron and the output is collected in the feature map.

> The feature map is the output of one filter applied to the previous layer.

**Feature map** visualization will provide insight into the internal representations for specific input for each of the **Convolutional** layers in the model.

### <u>**Strides:**</u>

- Stride is the number of pixels shifts over the input matrix. 

> *Stride denotes how many steps we are moving in each steps in convolution. By default it is one.*

- When the stride is 1 then we move the filters to 1 pixel at a time. When the stride is 2 then we move the filters to 2 pixels at a time and so on. 

The below figure shows convolution would work with a stride of 2.

​										![stride](/Users/nitishharsoor/Prop/CNN-trans/stride.gif)	

### <u>Padding</u>:

Sometimes filter does not fit perfectly fit the input image. We have two options:

1. Pad the picture with zeros (zero-padding) so that it fits
2. Drop the part of the image where the filter did not fit. This is called valid padding which keeps only valid part of the image.

> Padding is a process of adding zeros to the input matrix symmetrically

-  So padding is used:
  - To maintain the dimension of output as in input , we use padding. 
  - It refers to the amount of pixels added to an image when it is being processed by the kernel of a **CNN**
- Padding depends on the dimension of filter.

### <u>ReLU | Rectified **Linear** Unit:</u>

ReLU stands for Rectified Linear Unit for a non-linear operation. The output is ***ƒ(x) = max(0,x).***

**Why ReLU is important** : 

* ReLU’s purpose is to introduce non-linearity in our ConvNet. 
* The rectified linear **activation function** overcomes the vanishing gradient problem, allowing models to learn **faster** and **perform** **better**.
* Since, the real world data would want our ConvNet to learn would be non-negative linear values.

![ReLU](/Users/nitishharsoor/Prop/CNN-trans/relu.png)

* There are other non linear functions such as **tanh** or **sigmoid** that can also be used instead of ReLU. 

* Most of the data scientists use ReLU since performance wise ReLU is better than the other two.

### <u>Softmax / Logistic Layer</u>

- Softmax or Logistic layer is the last layer of CNN. 
- It resides at the end of FC layer.
-  **Logistic** is used for `binary classification` and 
- **softmax** is for `multi-classification`.

------

## 3. Building Blocks of CNN

There are **three types of layers** in a Convolutional Neural Network:

1. **Input layer** 
2. **Convolutional Layers (Convo + ReLU)** - Used to scan across images.
3. **Pooling Layers** - Used to downsample images(for lower resolutiion image).
4. **Fully-Connected Layers**- Dense layers.
5. **Softmax** - 
6. **Dropout Layer** - Used to add regularization.
7. **Output layer**

### 0. Input Layer

- Input layer in CNN should contain image data. Image data is represented by three dimensional matrix as we saw earlier. You need to reshape it into a single column. 
  - Suppose you have image of dimension 28 x 28 =784, you need to convert it into 784 x 1 before feeding into input. If you have “m” training examples then dimension of input will be (784, m).

### 1.  Convolutional Layers

Convo layer is sometimes called feature extractor layer because features of the image are get extracted within this layer. [Convolutional layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) are comprised of **filters** and **feature maps.**

  1. Number of filters.
  2. Feature map.
  3. Filter Size.
  4. Stride.
  5. Padding.
  6. [Activation Function/Non-Linearity](### ReLU | Rectified Linear Unit:).

- First of all, a part of image is connected to Convo layer to perform convolution operation as we saw earlier and calculating the dot product between receptive field(it is a local region of the input image that has the same size as that of filter) and the filter. 
- Result of the operation is single integer of the output volume. Then we slide the filter over the next receptive field of the same input image by a Stride and do the same operation again. 
- We will repeat the same process again and again until we go through the whole image. The output will be the input for the next layer.

> Convo layer also contains ReLU activation to make all negative value to zero.

> *The primary purpose for a convolutional layer is to detect features such as edges, lines, blobs of color, and other visual elements. The filters can detect these features. The more filters that we give to a convolutional layer, the more features it can detect.*

### 2. Pooling layer

- Pooling layer is used to reduce the spatial volume of input image after convolution.

> 📌. It is used between two convolution layer.
>
> **⚠️ Warning:** If we apply Dense layer after Convo layer without applying pooling or max pooling, then it will be computationally expensive and we don’t want it. 

​                            ![pooling](/Users/nitishharsoor/Prop/CNN-trans/pooling.jpeg)

 ***So, the max pooling is only way to reduce the spatial volume of input image.***

In the above example, we have applied max pooling in single depth slice with Stride of 2. You can observe the 4 x 4 dimension input is reduce to 2 x 2 dimension.

There is no parameter in pooling layer but it has two hyperparameters — Filter(F) and Stride(S).

### 3. Dense layer | Fully connected layers

- Fully connected layer involves weights, biases, and neurons. 
- It connects neurons in one layer to neurons in another layer. It is used to classify images between different category by training.
- Only after pooling we flattened our matrix into vector and feed it into a fully connected layer like a neural network.

![dense](/Users/nitishharsoor/Prop/CNN-trans/dense.png)

In the above diagram, the feature map matrix will be converted as vector (x1, x2, x3, x4). With the fully connected layers, we combined these features together to create a model. Finally, we have an activation function such as softmax or sigmoid to classify the outputs as cat | y1, dog | y2 , car | y3.

### 4. Softmax

- Softmax or Logistic layer is the last layer of CNN. It resides at the end of FC layer. Logistic is used for binary classification and softmax is for multi-classification.

## Setup of simple Convo_2D using Keras

In order to implement CNNs, most successful architecture uses one or more stacks of convolution + pool layers with relu activation, followed by a flatten layer then one or two dense layers.

​               **![connn](/Users/nitishharsoor/Prop/CNN-trans/connn.png)**

As we move through the network, feature maps become smaller spatially, and increase in depth. Features become increasingly abstract and lose spatial information. For example, the network understands that the image contained an eye, but it is not sure where it was.

Here’s an example of a typical CNN network in Keras.

![cnn_keras](/Users/nitishharsoor/Prop/CNN-trans/cnn_keras.png)

Here’s the result when you do `model.summary()`

![cnn-summary](/Users/nitishharsoor/Prop/CNN-trans/cnn-summary.png)

### Conv2d_1

![summar1](/Users/nitishharsoor/Prop/CNN-trans/summar1.png)

`Filter size ( 3 x 3) * input depth (1) * # of filters (32) + Bias 1/filter (32) = 320.` Here, the input depth is 1, because it’s for MNIST black and white data. Note that in tensorflow by default every convolution layer has bias added. 	

### Max_pooling2d_1

![summary2](/Users/nitishharsoor/Prop/CNN-trans/summary2.png)

Pooling layers don’t have parameters

### Conv2d_2

![summary3](/Users/nitishharsoor/Prop/CNN-trans/summary3.png)

Filter size (3 x 3) * input depth (32) * # of filters (64) + Bias, 1 per filter (64) = 18496

### Flatten_1

 ![png](/Users/nitishharsoor/Prop/CNN-trans/sumarry4.png)

It unstacks the volume above it into an array.
![array](/Users/nitishharsoor/Prop/CNN-trans/array.png)
 ### Dense_1

![dense-1](/Users/nitishharsoor/Prop/CNN-trans/dense-1.png)Input Dimension (128) * Output 

Dimension (10) + One bias per output neuron (10) = 1290

##   Summary

Convolutional Neural Network (CNN) is a class of deep neural network (DNN) which is widely used for computer vision or NLP. During the training process, the network’s building blocks are repeatedly altered in order for the network to reach optimal performance and to classify images and objects as accurately as possible.

- Provide input image into convolution layer
- Choose parameters, apply filters with strides, padding if requires. Perform convolution on the image and apply ReLU activation to the matrix.
- Perform pooling to reduce dimensionality size
- Add as many convolutional layers until satisfied
- Flatten the output and feed into a fully connected layer (FC Layer)
- Output the class using an activation function (Logistic Regression with cost functions) and classifies images.

## Convolutional Neural Networks Best Practices

Now that we know about the building blocks for a convolutional neural network and how the layers hang together, we can review some best practices to consider when applying them.

- **Input Receptive Field Dimensions**: The default is 2D for images, but could be 1D such as for words in a sentence or 3D for video that adds a time dimension.
- **Receptive Field Size**: The patch should be as small as possible, but large enough to “see” features in the input data. It is common to use 3×3 on small images and 5×5 or 7×7 and more on larger image sizes.
- **Stride Width**: Use the default stride of 1. It is easy to understand and you don’t need padding to handle the receptive field falling off the edge of your images. This could increased to 2 or larger for larger images.
- **Number of Filters**: Filters are the feature detectors. Generally fewer filters are used at the input layer and increasingly more filters used at deeper layers.
- **Padding**: Set to zero and called zero padding when reading non-input data. This is useful when you cannot or do not want to standardize input image sizes or when you want to use receptive field and stride sizes that do not neatly divide up the input image size.
- **Pooling**: Pooling is a destructive or generalization process to reduce overfitting. Receptive field is almost always set to to 2×2 with a stride of 2 to discard 75% of the activations from the output of the previous layer.
- **Data Preparation**: Consider standardizing input data, both the dimensions of the images and pixel values.
- **Pattern Architecture**: It is common to pattern the layers in your network architecture. This might be one, two or some number of convolutional layers followed by a pooling layer. This structure can then be repeated one or more times. Finally, fully connected layers are often only used at the output end and may be stacked one, two or more deep.
- **Dropout**: CNNs have a habit of overfitting, even with pooling layers. Dropout should be used such as between fully connected layers and perhaps after pooling layers.

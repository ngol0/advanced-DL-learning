# Advanced Deep Learning with Tensorflow - Courses by DeepLearning.ai
## Introduction
These are the projects completed at the end of each module in two DeepLearning.AI courses by Andrew Ng: [Advanced Computer Vision](https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow/) and [Deep Generative Modelling](https://www.coursera.org/learn/generative-deep-learning-with-tensorflow/). 

Each course consists of 4 modules, with one project per module. The project details for relevant topics could be found below. The topics and models covered in each module include:

* **Object Detection:** model architecture of R-CNN, Fast R-CNN, Faster R-CNN, finetune RetinaNet.

* **Image Segmentation:** model architecture of U-Net, segmentation implemention with FCN
  
* **Visualization and Interpretability:** visualize model predictions and understand CNNs

* **Deep Generative Modelling:** style transfer, VAEs and GANs

## Project Details
These projects explore advanced computer vision techniques and generative deep learning models. Each project focuses on applying a specific model or technique to a practical dataset.

### Style Transfer:
*  *Main Objective:* Implement neural style transfer using the Inception model as feature extractor.

*  *Resutls:* A new image of the original dog (left) with the style of the right image.

  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/generative-models/original.png" width="600" title="one-frame2">
  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/generative-models/doggo.png" width="600" title="one-frame2">

### VAEs:
*  *Main Objective:* Train a Variational Autoencoder (VAE) using the anime faces dataset by MckInsey666. Then use this model to generate a gallery of anime faces.

*  *Resutls:*
   <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/generative-models/vae.png" width="600" title="one-frame2">

### GANs:
*  *Main Objective:* 

*  *Resutls:* 
  
### Object Detection:
*  *Main Objective:* Retrain **RetinaNet** to spot Zombies using just 5 training images.
  
*  *Specific Tasks:* Setup the model to restore pretrained weights and fine tune the classification layers.
  
*  *Results:* The boxes my model generated match 99.58% of the ground truth boxes with a relative tolerance of 0.3.
  
  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/advanced-computer-vision/exe2.jpg" width="600" title="one-frame2">
  
### Image Segmentation:
*  *Main Objective:* Build a model that predicts the segmentation masks (pixel-wise label map) of handwritten digits. This model will be trained on the M2NIST dataset, a multi digit MNIST.
  
*  *Specific Tasks:*  Build a Convolutional Neural Network (CNN) from scratch for the downsampling path and use a Fully Convolutional Network, FCN-8, to upsample and produce the pixel-wise label map. The model will be evaluated using the intersection over union (IOU) and Dice Score. 
  
*  *Results:* average IOU score of 75% when compared against the true segments.
  
  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/advanced-computer-vision/exe3.png" width="600" title="one-frame2">
  
### Visualization and Interpretability: (on-going)

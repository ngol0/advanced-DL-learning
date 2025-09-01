# Advanced Deep Learning with Tensorflow - Courses by DeepLearning.ai
## Introduction
This is the projects done after every module in the 2 courses provided by Andrew Ng's DeepLearning.ai, including: [Advanced Computer Vision](https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow/) and [Deep Generative Modelling](https://www.coursera.org/learn/generative-deep-learning-with-tensorflow/). Each course has 4 projects in total for 4 different topics (so 8 in total for 2 courses). The project details for relevant topics could be found below. The topics and models covered in each module include:

* #### Transfer Learning and Style Learning:
  - transfer learning with ResNet50, object localization model architecture, evaluating object localization. 

* #### Object Detection:
  model architecture of R-CNN, Fast R-CNN, Faster R-CNN, finetune RetinaNet.

* #### Image Segmentation:
  model architecture of U-Net, segmentation implemention with FCN

* #### Visualization and Interpretability:
  visualize model predictions and understand CNNs

* #### Deep Generative Modelling:
  VAEs and GANs

## Project Details
Projects that explore computer vision techniques and generative deep learning models. 

* #### VAEs:
  *Main Objective:* Train a Variational Autoencoder (VAE) using the anime faces dataset by MckInsey666. Then use this model to generate a gallery of anime faces.

  *Resutls:* 
  
* #### Object Detection:
  *Main Objective:* Retrain **RetinaNet** to spot Zombies using just 5 training images.
  
  *Specific Tasks:* Setup the model to restore pretrained weights and fine tune the classification layers.
  
  *Results:* The boxes my model generated match 99.58% of the ground truth boxes with a relative tolerance of 0.3.
  
  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/gif_frame_236.jpg" width="900" title="one-frame2">
  
* #### Image Segmentation:
  *Main Objective:* Build a model that predicts the segmentation masks (pixel-wise label map) of handwritten digits. This model will be trained on the M2NIST dataset, a multi digit MNIST.
  
  *Specific Tasks:*  Build a Convolutional Neural Network (CNN) from scratch for the downsampling path and use a Fully Convolutional Network, FCN-8, to upsample and produce the pixel-wise label map. The model will be evaluated using the intersection over union (IOU) and Dice Score. 
  
  *Results:* average IOU score of 75% when compared against the true segments.
  
  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/exe3.png" width="900" title="one-frame2">
  
* #### Visualization and Interpretability: (on-going)

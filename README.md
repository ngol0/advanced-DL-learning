# Advanced Computer Vision with Tensorflow - A course from DeepLearning.ai
## Introduction
This is the projects done after every module in the course provided by Andrew Ng's [DeepLearning.ai](https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow/). There are 4 projects in total for 4 different topics. The project details could be found below. The topics and models covered in each module include:

* #### Transfer Learning:
  transfer learning with ResNet50, object localization model architecture, evaluating object localization. 

* #### Object Detection:
  model architecture of R-CNN, Fast R-CNN, Faster R-CNN, finetune RetinaNet. \n

* #### Image Segmentation:
  model architecture of U-Net, segmentation implemention with FCN

* #### Visualization and Interpretability:
  visualize model predictions and understand CNNs


## Project Details
* #### Transfer Learning:
  *Objective:* uild a model to predict bounding boxes around images.

  *Task:* Use transfer learning on any of the pre-trained models available in Keras. Dataset: Caltech Birds - 2010 dataset.

  *Resutls:* iou score greater than 0.5 for 49.20% of the images.
  
* #### Object Detection:
  *Objective:* Retrain RetinaNet to spot Zombies using just 5 training images.
  
  *Task:* Setup the model to restore pretrained weights and fine tune the classification layers.
  
  *Results:* The boxes my model generated match 99.58% of the ground truth boxes with a relative tolerance of 0.3.
  
  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/gif_frame_189.jpg" width="900" title="one-frame">
  <img src="https://github.com/ngol0/advanced-computer-vision-learning/blob/main/gif_frame_236.jpg" width="900" title="one-frame2">
  
* #### Image Segmentation: (on-going)
  
* #### Visualization and Interpretability: (on-going)

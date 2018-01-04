# unet_keras_tensorboard

Implementation of Unet on Keras with BatchNormalization and Tensorboard visualization. 
The code is written keeping in mind that it will be run on Kaggle Carvana Image challenge (https://www.kaggle.com/c/carvana-image-masking-challenge).


After running Unet 4 times(epoch = 4) on only 12 images, we get the following output - 

left:image,
middle:ground truth,
right:unet segmentation output

![](images/tensorboard.png?raw=true)


Below are the losses at every batch, epoch for training and validation sets.

<p align="center">
<img src="https://github.com/YadavKapil/unet_keras_tensorboard/blob/master/images/g1.png" width="200"> <img src="https://github.com/YadavKapil/unet_keras_tensorboard/blob/master/images/g1.png" width="200">
</p>


References - 
1. Kaggle Discussions / kernels
2. Other Unet implementations of Unet on Github like - 
3. Tensorboard implementation for logging at - 

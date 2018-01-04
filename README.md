# unet_keras_tensorboard

Implementation of Unet on Keras with BatchNormalization and Tensorboard visualization. 
The code is written keeping in mind that it will be run on Kaggle Carvana Image challenge (https://www.kaggle.com/c/carvana-image-masking-challenge).


After running Unet 4 times(epoch = 4) on only 12 images, we get the following output - 

left:image, middle:ground truth, right:unet segmentation output

![Alt-text-1](images/tensorboard.png?raw=true segmentation output) ![Alt-text-2](images/tensorboard.png?raw=true segmentation output)

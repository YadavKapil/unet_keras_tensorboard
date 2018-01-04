#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:51:19 2017

@author: Kapil Yadav (kapilsemailid@gmail.com)
"""
from __future__ import division, print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image 

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, core, BatchNormalization
from keras import losses
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

PROJECT_PATH_1 = '/Users/z001lg8/Documents/Kapil_docs_personal/Kaggle/carvana/'
PROJECT_PATH_2 = '/home_dir/z001lg8/kg/carvana/'
im_size=(160,160)

if os.path.isdir('/Users/z001lg8/'):
	sys.path.append(PROJECT_PATH_1 + 'code')
	project_path = PROJECT_PATH_1
else: 
	sys.path.append(PROJECT_PATH_2 + 'code')
	project_path = PROJECT_PATH_2

from tensorboard_unet import TensorBoardNew

from keras.preprocessing.image import ImageDataGenerator
import pkg_resources as p
p.get_distribution('keras')

from keras import backend as K
K.backend()
K.set_image_data_format('channels_last')




def create_log_dir():
	f=open(project_path+'code/logs/log_count.txt')
	count=int(f.readlines()[0])
	log_dir=project_path+'code/logs/log'+str(count)
	os.mkdir(log_dir)
	f = open(project_path+'code/logs/log_count.txt', "w")
	f.write(str(count+1))
	f.close()
	return log_dir

def read_image_and_mask(data_prefix,car_id,angle):
	data = []
	im = cv2.cvtColor(cv2.imread(project_path+'data_and_results/{}/{}_{}.jpg'.format(data_prefix, car_id, angle)), cv2.COLOR_BGR2RGB)
	data.append(im)
	if data_prefix=='train':
		mask=Image.open(project_path+'data_and_results/{}_masks/{}_{}_mask.gif'.format(data_prefix,car_id,angle))
		mask=np.array(mask.getdata()).reshape(im.shape[:2])
		data.append(mask.astype(np.uint8))
	return data

def resize_data(data):
	data[0]=cv2.resize(data[0],None,fx=0.25,fy=0.25,interpolation = cv2.INTER_AREA)
	data[1]=cv2.resize(data[1],None,fx=0.25,fy=0.25,interpolation = cv2.INTER_AREA)
	#return data

smooth=1.
def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

def categorical_crossentropy_with_logit(y_true, y_pred):
	return K.categorical_crossentropy(y_true,y_pred,from_logits=True)

def conv_relu_bn(prev_layer,filters,kernel_size=(3,3),axis=-1,momentum=0.99,center=True,scale=False):
	cnv = Conv2D(filters=filters,kernel_size=kernel_size,activation='relu',padding='same')(prev_layer)
	cnv_bn = BatchNormalization(axis=axis,momentum=momentum,center=center,scale=scale)(cnv)
	return cnv_bn

def unet_model(im_shape):
	inputs=Input(shape=im_shape)
	
	conv1_1=conv_relu_bn(inputs,64)
	conv1_2=conv_relu_bn(conv1_1,64)
	pool1=MaxPooling2D((2,2))(conv1_2)
	
	conv2_1=conv_relu_bn(pool1,128)
	conv2_2=conv_relu_bn(conv2_1,128)
	pool2=MaxPooling2D((2,2))(conv2_2)
	
	conv3_1=conv_relu_bn(pool2,256)
	conv3_2=conv_relu_bn(conv3_1,256)
	pool3=MaxPooling2D((2,2))(conv3_2)
	
	conv4_1=conv_relu_bn(pool3,512)
	conv4_2=conv_relu_bn(conv4_1,512)
	
	up5=concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv3_2], axis=3)
	conv5_1=conv_relu_bn(up5,256)
	conv5_2=conv_relu_bn(conv5_1,256)
	
	up6=concatenate([UpSampling2D(size=(2, 2))(conv5_2), conv2_2], axis=3)
	conv6_1=conv_relu_bn(up6,128)
	conv6_2=conv_relu_bn(conv6_1,128)
	
	up7=concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv1_2], axis=3)
	conv7_1=conv_relu_bn(up7,64)
	conv7_2=conv_relu_bn(conv7_1,64)
	
	conv8=conv_relu_bn(prev_layer=conv7_2,filters=2,kernel_size=(1,1),scale=True)
	#out=sft(conv8,-1)
	out=core.Activation('softmax')(conv8)
	
	model=Model(inputs=inputs,outputs=out)
	return model

def unet_model_without_bn(im_shape):
	inputs=Input(shape=im_shape)
	
	conv1_1=Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(inputs)
	conv1_2=Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(conv1_1)
	pool1=MaxPooling2D((2,2))(conv1_2)
	
	conv2_1=Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(pool1)
	conv2_2=Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(conv2_1)
	pool2=MaxPooling2D((2,2))(conv2_2)
	
	conv3_1=Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(pool2)
	conv3_2=Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(conv3_1)
	pool3=MaxPooling2D((2,2))(conv3_2)
	
	conv4_1=Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same')(pool3)
	conv4_2=Conv2D(filters=512,kernel_size=(3,3),activation='relu',padding='same')(conv4_1)
	
	up5=concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv3_2], axis=3)
	conv5_1=Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(up5)
	conv5_2=Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same')(conv5_1)
	
	up6=concatenate([UpSampling2D(size=(2, 2))(conv5_2), conv2_2], axis=3)
	conv6_1=Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(up6)
	conv6_2=Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same')(conv6_1)
	
	up7=concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv1_2], axis=3)
	conv7_1=Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(up7)
	conv7_2=Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(conv7_1)
	
	conv8=Conv2D(filters=2,kernel_size=(1,1),activation='relu',padding='same')(conv7_2)
	out=core.Activation('softmax')(conv8)
	
	model=Model(inputs=inputs,outputs=out)
	return model

def display_im_and_mask(a):
	plt.close('all')
	fig1 = plt.figure(0)
	fig1.add_subplot(3,3,1)
	plt.imshow(a[0][0].astype(np.uint8),vmin=0,vmax=255)
	
	fig1.add_subplot(3,3,2)
	plt.imshow(a[1][0,...,0].astype(np.uint8),vmin=0,vmax=255,cmap='gray')
	
	fig1.add_subplot(3,3,3)
	plt.imshow(a[1][0,...,1].astype(np.uint8),vmin=0,vmax=255,cmap='gray')
	
	fig1.add_subplot(3,3,4)
	plt.imshow(a[0][1].astype(np.uint8),vmin=0,vmax=255)
	
	fig1.add_subplot(3,3,5)
	plt.imshow(a[1][1,...,0].astype(np.uint8),vmin=0,vmax=255,cmap='gray')
	
	fig1.add_subplot(3,3,6)
	plt.imshow(a[1][1,...,1].astype(np.uint8),vmin=0,vmax=255,cmap='gray')
	
	fig1.add_subplot(3,3,7)
	plt.imshow(a[0][2].astype(np.uint8),vmin=0,vmax=255)
	
	fig1.add_subplot(3,3,8)
	plt.imshow(a[1][2,...,0].astype(np.uint8),vmin=0,vmax=255,cmap='gray')
	
	fig1.add_subplot(3,3,9)
	plt.imshow(a[1][2,...,1].astype(np.uint8),vmin=0,vmax=255,cmap='gray')

def combine_generator(gen1, gen2):
	while True:
		msk=gen2.next()
		msk_neg=255.-msk
		mask_concat=np.concatenate((msk,msk_neg),axis=3)
		yield(gen1.next(),mask_concat)


def get_generator(args,split_type,batch_size):
	image_datagen = ImageDataGenerator(**args)
	mask_datagen = ImageDataGenerator(**args)
	
	# Provide the same seed and keyword arguments to the fit and flow methods
	seed = 1
	
	image_generator = image_datagen.flow_from_directory(
		project_path+'data_and_results/unet_training_data/unet_{}_im'.format(split_type),
		target_size=im_size,
		batch_size=batch_size,
		class_mode=None,
		seed=seed)
	
	mask_generator = mask_datagen.flow_from_directory(
		project_path+'data_and_results/unet_training_data/unet_{}_mask_png'.format(split_type),
		target_size=im_size,
		batch_size=batch_size,
		color_mode='grayscale',
		class_mode=None,
		seed=seed)
	
	# combine generators into one which yields image and masks
	generator = combine_generator(image_generator, mask_generator)
	return generator




tr_batch_size=3
val_batch_size=3

data_gen_args = dict()
train_generator=get_generator(data_gen_args,'train',tr_batch_size)
val_generator=get_generator(data_gen_args,'val',val_batch_size)

log_dir=create_log_dir()
log_dir

K.clear_session()
K.set_learning_phase(1) #set learning phase

model=unet_model(im_size+(3,))
model.compile(optimizer=Adam(lr=0.0001), loss=losses.categorical_crossentropy, metrics=[dice_coef,losses.categorical_crossentropy])
tsb=TensorBoardNew(log_dir=log_dir,batch_to_skip=3,batch_to_skip_logs=3,steps_per_epoch=4,batch_size=val_batch_size, write_graph=True, write_grads=True, write_images=True,val_gen=val_generator)

model_checkpoint = ModelCheckpoint(project_path+'code/weights.h5', monitor='val_loss', save_best_only=True)

print('log_dir:',log_dir)

model.fit_generator(
	train_generator,
	validation_data=val_generator,
	validation_steps=1,
	steps_per_epoch=4,
	epochs=5, verbose=1,
	callbacks=[model_checkpoint,tsb],
	initial_epoch=0)

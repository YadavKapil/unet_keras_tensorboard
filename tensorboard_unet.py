#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 06:43:10 2017

@author: z001lg8
"""


from __future__ import absolute_import
from __future__ import print_function

from keras import backend as K
from keras.callbacks import Callback

if K.backend() == 'tensorflow':
	import tensorflow as tf




class TensorBoardNew(Callback):
	
	def __init__(self, log_dir='./logs',
					batch_to_skip=0,
					batch_to_skip_logs=0,
					steps_per_epoch=4,
					batch_size=3,
					write_graph=True,
					write_grads=False,
					write_images=False,
					val_gen=None):
		super(TensorBoardNew, self).__init__()
		if K.backend() != 'tensorflow':
			raise RuntimeError('TensorBoard callback only works '
									'with the TensorFlow backend.')
		self.log_dir = log_dir
		self.batch_to_skip = batch_to_skip
		self.batch_to_skip_logs=batch_to_skip_logs
		self.merged = None
		self.write_graph = write_graph
		self.write_grads = write_grads
		self.write_images = write_images
		self.val_gen = val_gen
		self.epoch=0
		self.steps_per_epoch=steps_per_epoch
		self.batch_size=batch_size
		self.step=0
	
	def set_model(self, model):
		self.model = model
		self.sess = K.get_session()
		#tf.summary.scalar('batch_loss1',self.model.total_loss)
		
		tgt = self.grayscale_to_rgb(model.targets[0][...,:1])
		out = self.grayscale_to_rgb(model.output[...,:1])
		self.pred1=tf.concat([model.input[:1],tgt,out],0)
		
		self.writer = tf.summary.FileWriter(self.log_dir)
	
	def on_batch_end(self, batch, logs=None):
		self.step = self.epoch*self.steps_per_epoch + batch
		step=self.step
		loss = logs['loss']
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = loss
		summary_value.tag = 'batch_loss'
		self.writer.add_summary(summary, step)
		
		if step % self.batch_to_skip == 0 and self.val_gen is not None:
			print('inside. Writing Image')
			im_summ = tf.summary.image("prediction_%i"%self.step,self.pred1,3)
			
			inps = self.standardize_data(self.val_gen.next())
			tensors=(self.model.inputs + self.model.targets+self.model.sample_weights)
			feed_dict = dict(zip(tensors,inps))
			
			summary=self.sess.run(im_summ,feed_dict=feed_dict)
			
			self.writer.add_summary(summary,step)
		
		if step % self.batch_to_skip_logs == 0:
			logs = logs or {}
			for name, value in logs.items():
				if name in ['batch', 'size']:
					continue
				summary = tf.Summary()
				summary_value = summary.value.add()
				summary_value.simple_value = value.item()
				summary_value.tag = name
				self.writer.add_summary(summary, step)
			name = 'learning_rate'
			value = self.get_learning_rate()
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value
			summary_value.tag = name
			self.writer.add_summary(summary, step)
		self.writer.flush()
	
	def on_epoch_end(self, epoch, logs=None):
		self.epoch+=1
		logs = logs or {}
		for name, value in logs.items():
			if name in ['batch', 'size']:
				continue
			if 'val' not in name:
				continue
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, epoch)
			
			name = 'epochs'
			value = epoch
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value
			summary_value.tag = name
			self.writer.add_summary(summary, self.epoch*self.steps_per_epoch)

		self.writer.flush()
	
	def on_train_end(self, _):
		self.writer.close()
	
	def standardize_data(self,validation_data):
		if len(validation_data) == 2:
			val_x, val_y = validation_data
			val_sample_weight = None
		
		val_x, val_y, val_sample_weights = self.model._standardize_user_data(val_x, val_y, val_sample_weight)
		val_data = val_x + val_y + val_sample_weights
		if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
			val_data += [0.]
		return val_data
	def grayscale_to_rgb(self,tensor):
		return tf.concat([tensor,tensor,tensor],axis=3)
	
	def get_learning_rate(self):
		optimizer = self.model.optimizer
		lr = K.eval(optimizer.lr)
		if optimizer.initial_decay > 0:
			lr *= (1. / (1. + optimizer.decay * optimizer.iterations))
		
		t = optimizer.iterations + 1
		t = K.cast(t, dtype='float32')
		
		lr_t = lr * (K.sqrt(1. - K.pow(optimizer.beta_2, t)) /(1. - K.pow(optimizer.beta_1, t)))
		
		learning_rate = K.eval(lr_t)
		print('\nlearning rate:{:10f}'.format(learning_rate))
		return learning_rate

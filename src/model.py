from SpatialPyramidPooling import SpatialPyramidPooling

from keras.applications import vgg16
from keras.backend.tensorflow_backend import set_session

import numpy as np
import keras.backend as K
import tensorflow as tf

import os
import sys

from keras.models import Model
from keras.layers import (
    Input,
    concatenate,
    UpSampling2D,
    Dropout,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
)

network_state = ['train', 'ft', 'test']
class SaliencyUnet(object):
    def __init__(self, img_rows = None, img_cols = None, state='train'):
        self.img_rows = img_rows
        self.img_cols = img_cols
        if not state in network_state:
            raise Exception('NetWork state unknown.Only support "train","ft","test".')
        else:
            self.state = state
    def BuildModel(self):
        inputs = Input((self.img_rows, self.img_cols, 3))
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='block1_conv1')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='block1_conv2')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='block2_conv1')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='block2_conv2')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)
        #model_for_share = Model(inputs, pool2)
        #if self.state == 'train':
            #model_for_share.load_weights('../../data/model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='block3_conv1')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='block3_conv2')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block4_conv1')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block4_conv2')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block5_conv1')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block5_conv2')(conv5)
        up6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block6_conv1')\
            (UpSampling2D(size = (2,2), name='upsampling_1')(conv5))
        merge6 = concatenate([conv4,up6], axis = -1, name='concat_1')
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block6_conv2')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block6_conv3')(conv6)
        up7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block7_conv1')\
            (UpSampling2D(size = (2,2), name='upsampling_2')(conv6))
        merge7 = concatenate([conv3,up7], axis = -1, name='concat_2')
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block7_conv2')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block7_conv3')(conv7)
        up8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block8_conv1')\
            (UpSampling2D(size = (2,2), name='upsampling_3')(conv7))
        merge8 = concatenate([conv2,up8], axis = -1, name='concat_3')
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block8_conv2')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block8_conv3')(conv8)
        up9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block9_conv1')\
            (UpSampling2D(size = (2,2), name='upsampling_4')(conv8))
        merge9 = concatenate([conv1,up9], axis = -1, name='concat_4')
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block9_conv2')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block9_conv3')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='unet_block9_conv4')(conv9)
        #conv10 = Conv2D(1, 1, activation = 'sigmoid', name='segmentation')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', name='segmentation')(conv9)
        model = Model(inputs = [inputs], outputs = [conv10])
        if self.state == 'train':
            model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
        return model

class ofn_net():

    def __init__(self, state='train'):

        self.weights = os.path.join('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        if not state in network_state:   
            raise Exception('NetWork state unknown.Only support "train","ft","test".')
        else:
            self.state = state

    def set_model(self):
        #roi_input = Input(shape=(4,))
        #model = vgg16.VGG16(include_top=False, weights=None, input_shape=(None, None, 3))
        model_vgg = vgg16.VGG16(include_top=False, weights=None, input_shape=(None, None, 3))
        for i in model_vgg.layers:
            i.trainble = False
        if self.state == 'train':
            model_vgg.load_weights(self.weights)

        x = SpatialPyramidPooling(pool_list=[1, 2, 4])(model_vgg.layers[-1].output)
        #x = Activation('relu')(x)
        #x = Flatten(name='flatten')(x)
        #x = Flatten(name='flatten')(model.output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        #x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        #x = Dropout(0.5)(x)
        out_1 = Dense(4, activation='linear', name='prediction')(x)

        #self.model = Model([model.input, roi_input], out_1)
        model = Model(model_vgg.input, out_1)
        if self.state == 'ft':
            model.load_weights(self.weights)
        return model

    def set_weights(self, weight):
        self.weights = weight
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: msharrock
# version: 0.0.1

"""
VNet Blocks for DeepBleed  

tensorflow version 2.0

"""

import tensorflow as tf
from tensorflow.keras import layers


class VNetInBlock(layers.Layer):
    def __init__(self):
        super(VNetInBlock, self).__init__()
        self.add = layers.Add()
        self.concatenate = layers.Concatenate() 
        self.convolution = layers.Conv3D(filters=16, kernel_size=(5,5,5), strides=1, 
                                         padding='same', kernel_initializer='he_normal', activation='relu') 

    def call(self, inputs): 
        x = self.convolution(inputs)
        d = self.concatenate(16 * [inputs])

        return self.add([x, d])


class VNetDownBlock(layers.Layer):
    def __init__(self, channels, n_convs):
        super(VNetDownBlock, self).__init__()
        self.channels = channels
        self.n_convs = n_convs
        self.add = layers.Add()
        self.downsample = layers.Conv3D(filters=self.channels, kernel_size=(2,2,2), strides=2,
                                         padding='valid', kernel_initializer='he_normal', activation='relu')
        self.convolution = layers.Conv3D(filters=self.channels, kernel_size=(5,5,5), strides=1, 
                                         padding='same', kernel_initializer='he_normal', activation='relu') 

    def call(self, inputs):  
        d = self.downsample(inputs) 
        
        for _ in range(self.n_convs):
            x = self.convolution(d)
            
        return self.add([x, d])  

class VNetUpBlock(layers.Layer):
    def __init__(self, channels, n_convs):
        super(VNetUpBlock, self).__init__()
        self.channels = channels
        self.n_convs = n_convs
        self.add = layers.Add() 
        self.concatenate = layers.Concatenate() 
        self.upsample = layers.Conv3DTranspose(filters=self.channels//2, kernel_size=(2,2,2), strides=2,
                                               padding='valid', kernel_initializer='he_normal', activation='relu')
        self.convolution = layers.Conv3D(filters=self.channels, kernel_size=(5,5,5), strides=1, 
                                         padding='same', kernel_initializer='he_normal', activation='relu') 

    def call(self, inputs, skip):  
        x = self.upsample(inputs)
        cat = self.concatenate([x, skip])
        
        for _ in range(self.n_convs):
            x = self.convolution(cat)
                   
        return self.add([x, cat])  


class VNetOutBlock(layers.Layer):

    def __init__(self):
        super(VNetOutBlock, self).__init__()             
        self.final = layers.Conv3D(filters=2, kernel_size=(1,1,1), strides=1, 
                                         padding='valid', kernel_initializer='he_normal', activation='relu')
        
        self.binary = layers.Conv3D(filters=1, kernel_size=(1,1,1), strides=1, 
                                         padding='valid', kernel_initializer='he_normal', activation='sigmoid')
               
    def call(self, inputs):     
        x = self.final(inputs)

        return self.binary(x)


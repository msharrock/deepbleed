
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: msharrock
# version: 0.0.1

"""
Neural Net Models for DeepBleed 

tensorflow version 2.0

"""

import tensorflow as tf
from tensorflow.keras import layers
from blocks.vnet import VNetDownBlock, VNetUpBlock, VNetInBlock, VNetOutBlock

"""
Model below is the VNet architecture for volumetric anatomic segmentation, 
originally by Milletari et al.

'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation'

https://arxiv.org/abs/1606.04797


"""


class VNet(keras.Model):
    def __init__(self):
        super(VNet, self).__init__()
        input_layer = VNetInBlock(16)
        down_1 = VNetDownBlock(32, 2)
        down_2 = VNetDownBlock(64, 3)
        down_3 = VNetDownBlock(128, 3)
        down_4 = VNetDownBlock(256, 3)
        up_4 = VNetUpBlock(256, 3)
        up_3 = VNetUpBlock(128, 3)
        up_2 = VNetUpBlock(64, 2)
        up_1 = VNetUpBlock(32, 1)
        outblock = VNetOutBlock(32)
        
    def call(self, inputs, shape):
        inputs = layers.Input(shape = shape)
        x_16 = input_layer(inputs) 
        x_32 = down_1(x_16)     
        x_64 = down_2(x_32) 
        x_128 = down_3(x_64)
        x_256 = down_4(x_128) 
        
        x = up_4(x_256, skip=x_128)
        x = up_3(x, skip=x_64)
        x = up_2(x, skip=x_32)
        x = up_1(x, skip=x_16)
        outputs = outblock(x)

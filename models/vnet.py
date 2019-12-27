
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: msharrock
# version: 0.0.1

"""
Neural Net Models for DeepBleed 

tensorflow version: 2.0.0-beta1

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


class VNet(tf.keras.Model):
    def __init__(self):
        super(VNet, self).__init__()
        self.input_layer = VNetInBlock()
        self.down_1 = VNetDownBlock(32, 2)
        self.down_2 = VNetDownBlock(64, 3)
        self.down_3 = VNetDownBlock(128, 3)
        self.down_4 = VNetDownBlock(256, 3)
        self.up_4 = VNetUpBlock(256, 3)
        self.up_3 = VNetUpBlock(128, 3)
        self.up_2 = VNetUpBlock(64, 2)
        self.up_1 = VNetUpBlock(32, 1)
        self.outblock = VNetOutBlock()
        
    def call(self, inputs):
        x_16 = self.input_layer(inputs) 
        x_32 = self.down_1(x_16)     
        x_64 = self.down_2(x_32) 
        x_128 = self.down_3(x_64)
        x_256 = self.down_4(x_128) 
        
        x = self.up_4(x_256, skip=x_128)
        x = self.up_3(x, skip=x_64)
        x = self.up_2(x, skip=x_32)
        x = self.up_1(x, skip=x_16)
        
        return self.outblock(x)
               
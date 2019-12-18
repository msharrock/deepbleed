#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: msharrock
# version: 0.0.1

"""
Image Format Conversion for DeepBleed 

"""
import os
import ants
import nibabel as nib
import tensorflow as tf

def ants2nii(image):
    array_data = image.numpy()
    affine = np.hstack([image.direction*np.diag(image.spacing),np.array(image.origin).reshape(3,1)])
    affine = np.vstack([affine, np.array([0,0,0,1.])])
    nii_image = nib.Nifti1Image(array_data, affine)
    return nii_image

def nii2ants(image):
    from tempfile import mktemp
    tmpfile = mktemp(suffix='.nii.gz')
    image.to_filename(tmpfile)
    ants_image = ants.image_read(tmpfile, pixeltype = 'float')
    os.remove(tmpfile)
    return ants_image

def ants2tf(image):
    ants_params = [image.origin, image.spacing, image.direction]
    image = image.numpy()
    tf_image = tf.convert_to_tensor(image, dtype=tf.float32)
    tf_image = tf.expand_dims(tf_image, -1)
    return tf_image, ants_params

def tf2ants(image, ants_params):
    image = tf.squeeze(image)
    image = image.numpy()
    return image


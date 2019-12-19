#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: msharrock
# version: 0.0.1

"""
Prediction Script for DeepBleed 

Command Line Arguments:
        --indir: string, location to perform prediction
        --outdir: string, location to save predictions
        --weights: string, optional, location of model weights
        --cpus: int, optional, number of cpu cores to utilize
        --gpus: int, optional, number of gpus to utilize

"""

import os
import shutil

import fsl
import ants 
import nibabel as nib
import tensorflow as tf

from tools import parse
from preprocess import extract, register, convert
from models import vnet

# load command line arguments
setup = parse.args('predict')
if setup.CPUS == None:
    setup.CPUS = '1'
# environmental variable setup
os.environ["ANTS_RANDOM_SEED"] = '1'
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

if setup.CPUS == None:
    pass
else:
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = setup.CPUS

if setup.GPUS == None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(setup.GPUS)))

# set paths to scripts, templates, weights etc. 
TEMPLATE_PATH = os.path.join('templates', 'scct_unsmooth_SS_0.01_128x128x128.nii.gz')
WEIGHT_PATH = setup.weights

# setup directory trees
IN_DIR = setup.IN_DIR
OUT_DIR = setup.OUT_DIR 

# load the model and weights
model = VNet()
model.load_weights(WEIGHT_PATH)

# load input data
files = sorted(next(os.walk(IN_DIR)))[2]
files = [os.path.join(IN_DIR, f) for f in files]
template = ants.image_read(TEMPLATE_PATH, pixeltype = 'float')

for filename in files:

# preprocessing    
    image = nib.load(filename)
    image = extract.brain(image)
    image = convert.nii2ants(image)
    image, transforms = register.rigid(template, image)
    image = convert.ants2np(image)

# neural net prediction    
    prediction = model.predict(image, batch_size=1, workers=setup.CPUS)

# invert registration
    original_image = ants.image_read(filename)
    prediction = convert.np2ants(prediction)
    prediction = register.invert(image, prediction, transforms)
    prediction = convert.ants2nii(prediction)
    nib.save(prediction, os.path.join(OUT_DIR, filename))


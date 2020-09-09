# @author: msharrock
# version: 0.0.1

"""
Image Format Conversion for DeepBleed 

"""
import os
import ants
import numpy as np
import nibabel as nib
import tensorflow as tf

def ants2nii(image):
    array_data = image.numpy()
    affine = np.hstack([image.direction*np.diag(image.spacing),np.array(image.origin).reshape(3,1)])
    affine = np.vstack([affine, np.array([0,0,0,1.])])
    nii_image = nib.Nifti1Image(array_data, affine)
    return nii_image

def nii2ants(image):
    ndim = image.ndim
    q_form = image.get_qform()
    spacing = image.header["pixdim"][1 : ndim + 1]

    origin = np.zeros((ndim))
    origin[:3] = q_form[:3, 3]

    direction = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    image = ants.from_numpy(
        data = image.get_fdata(),
        origin = origin.tolist(),
        spacing = spacing.tolist(),
        direction = direction )
    return image

def ants2np(image):
    ants_params = [image.origin, image.spacing, image.direction]
    image = image.numpy().copy()
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)
    return image, ants_params

def np2ants(image, ants_params):
    image = np.squeeze(image)
    image = (image > 0.5).astype(np.float32)
    image = ants.from_numpy(image, origin = ants_params[0], spacing = ants_params[1], direction = ants_params[2])
    return image


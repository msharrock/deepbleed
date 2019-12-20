#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: msharrock
# version: 0.0.1

'''
Extraction methods for DeepBleed 
       
'''

import os

from fsl.wrappers import fslmaths, bet


def brain(image):

        '''
        Brain Extraction with FSL 

        Params:
        - image: nifti object, scan to brain extract
        Output: 
        - brain_image: nifti object, extracted brain
        '''
        affine = image.affine
        header = image.header
        tmpfile = 'tmpfile.nii.gz'
        image.to_filename(tmpfile)

        # FSL calls
        mask = fslmaths(image).thr('0.000000').uthr('100.000000').bin().fillh().run()
        fslmaths(image).mas(mask).run(tmpfile)
        bet(tmpfile, tmpfile, fracintensity = 0.01)
        mask = fslmaths(tmpfile).bin().fillh().run()
        image = fslmaths(image).mas(mask).run()
        image = nib.Nifti1Image(image.get_data(), affine, header)
        os.remove(tmpfile)

        return image


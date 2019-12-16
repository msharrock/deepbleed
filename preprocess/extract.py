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

        tmpfile = 'tmpfile.nii.gz'
        image.to_filename(tmpfile)
        mask = fslmaths(image).thr('0.000000').uthr('100.000000').bin().fillh().run()
        fslmaths(image).mas(mask).run(tmpfile)
        bet(tmpfile, tmpfile, fracintensity = 0.01)
        mask = fslmaths(tmpfile).bin().fillh().run()
        brain_image = fslmaths(image).mas(mask).run()
        os.remove(tmpfile)

        return brain_image


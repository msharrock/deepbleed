#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: msharrock
# version: 0.0.1

'''
Command Line Argument Parsing Tools

'''

import os
import argparse

def args(setting):

    '''
    Command Line Argument Interpreter

    Params:
        - session: string, 'predict','train','validate'

    Returns:
        - args: object, accessible representation of arguments
    '''
    if setting == 'predict':

        parser = argparse.ArgumentParser(description="Arguments for Prediction of Hemorrhage")
    
        parser.add_argument('--indir', required=True, action='store',
                            dest=IN_DIR, help='input directory')
        parser.add_argument('--outdir', required=True, action='store',
                            dest=OUT_DIR, help='output directory')
        parser.add_argument('--weights', required=False, action='store',
                            dest='weights', help='tf model weights')
        parser.add_argument('--gpus', required=False, action='store',
                            type=int, dest=GPUS, 
                            help='input system ID of GPU or input -1 for all')
        parser.add_argument('--cpus', required=False, action='store',
                            type=int, dest=CPUS, help='number of CPU cores')
    
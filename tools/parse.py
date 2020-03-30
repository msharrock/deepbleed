# @author: msharrock
# version: 0.0.1

'''
Command Line Argument Parsing Tools

'''

import os
import argparse

def args(mode):

    '''
    Command Line Argument Interpreter

    Params:
        - session: string, 'predict','train','validate'

    Returns:
        - args: object, accessible representation of arguments
    '''
    if mode == 'predict':

        parser = argparse.ArgumentParser(description="Arguments for Prediction of Hemorrhage")
    
        parser.add_argument('-i','--indir', required=True, action='store',
                            dest='IN_DIR', help='input directory')
        parser.add_argument('-o','--outdir', required=True, action='store',
                            dest='OUT_DIR', help='output directory')
        parser.add_argument('-w','--weights', required=True, action='store',
                            dest='weights', help='tf model weights')
        parser.add_argument('--gpus', required=False, action='store',
                            type=int, dest='GPUS', 
                            help='number of GPUs to utilize')
        parser.add_argument('--cpus', required=False, action='store',
                            type=int, dest='CPUS', help='number of CPU cores')
        parser.add_argument('-v', '--verbose', required=False, action='store_true',
                            help='verbose and timed script')
        parser.add_argument('-b', '--brain', required=False, action='store_true',
                            help='image is brain-only, extraction skipped')

    return parser.parse_args()
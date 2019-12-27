# @author: msharrock
# version: 0.0.1

"""
Registration methods for DeepBleed 

"""
import ants

def rigid(fixed, moving):

        '''
        Rigid Registration with ANTs

        Params:
                - moving: ants image, image to move when registering
                - fixed: ants image, template image to register to

        Outputs: 
                - image: registered image
                - transforms: transformation affine matrix
        '''

        kwargs = {'-n': 'nearestNeighbor'}
        tx = ants.registration(fixed, moving, type_of_transform='Rigid', mask=None, grad_step=0.2, flow_sigma=3, total_sigma=0, 
                           aff_metric='mattes', aff_sampling=64, syn_metric ='mattes',**kwargs) 
                        
        image = tx['warpedmovout']
        transforms = tx['fwdtransforms']
        return image, transforms


def invert(fixed, moving, transforms):
        '''
        Inverse Transform with ANTs

        Params:
                - image: ants image, image to revert
                - invtransform: affine matrix to use for inverse transform
        Outputs: 
                - image: ants image, inverted
        '''
        image = ants.apply_transforms(fixed = fixed, moving = moving, transformlist = transforms, interpolator = 'genericLabel', whichtoinvert = [True])   

        return image


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:11:14 2019

@author: aneesh
"""

import cv2
import random
from cv2 import ROTATE_90_CLOCKWISE
from cv2 import ROTATE_90_COUNTERCLOCKWISE
import numpy as np

class RandomHorizontallyFlip(object):
    '''
    Horizontally flips the image and corresponding mask randomly with a given probability

    Parameters:
        p (float) -- probability of the image being flipped. Default value is 0.5.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, img, hsi, mask):
        if random.random() < self.p:
            return (cv2.flip(img, 0), cv2.flip(hsi, 0), cv2.flip(mask, 0))
        return img, hsi, mask



class RandomVerticallyFlip(object):
    '''
    Vertically flips the image and corresponding mask randomly with a given probability

    Parameters:
        p (float) -- probability of the image being flipped. Default value is 0.5.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, img, hsi, mask):
        if random.random() < self.p:
            return (cv2.flip(img, 1), cv2.flip(hsi, 1), cv2.flip(mask, 1))
        return img, hsi, mask

class RandomTranspose(object):
    '''
    Shifts the image and corresponding mask by 90 deg randomly with a given probability

    Parameters:
        p (float) -- probability of the image being flipped. Default value is 0.5.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, img, hsi, mask):
        if random.random() < self.p:
            return (cv2.transpose(img), np.transpose(hsi, (1, 0, 2)), cv2.transpose(mask))
        return img, hsi, mask

        
class RandomRotate(object):
    '''
    Shifts the image and corresponding mask by 90 deg randomly with a given probability

    Parameters:
        p (float) -- probability of the image being flipped. Default value is 0.5.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self, img, hsi, mask):
        if random.random() < self.p:
            return (cv2.rotate(img,ROTATE_90_COUNTERCLOCKWISE), np.rot90(hsi), cv2.rotate(mask,ROTATE_90_COUNTERCLOCKWISE))
        return img, hsi, mask

class Compose(object):
    '''
    Composes all the transforms together
    
    Parameters:
        augmentations -- list of all augmentations to compose
    '''
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, hsi, mask):
        for a in self.augmentations:
            img, hsi, mask = a(img, hsi, mask)
        
        return img, hsi, mask
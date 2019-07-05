from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.misc
import random
import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import imageio
from skimage import io
import augmentation_transforms_ as at
from os import listdir
from os.path import isfile, isdir, join

mypath = "Path"
files = listdir(mypath)
for f in files:
    top = np.random.randint(0,10)
    fullpath = join(mypath, f)
    new=mypath+'/'+f
    print(new)
    img= Image.open(new)
    img_array = np.array(img)
    #data augmentation
    if top>2:
        newimg=ImageOps.equalize(img)
    else:
        newimg=img
    if top>4:
        newimg=ImageOps.equalize(newimg)
    newname='image_path'+f
    scipy.misc.imsave(newname,newimg)
    print ('-------------------------------------')

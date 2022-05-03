
from cProfile import label
import os
import os.path as osp
import numpy as np

from skimage import io
from PIL import Image
from collections import Counter
folder_dir = osp.join('Aerial Data', 'Collection') #path to full files
labels = io.imread(osp.join(folder_dir, 'image_labels.tif'))[53:,7:,:]
# print(labels)
# for label in labels:
    # print(label)
classes = [[  0 ,  0 ,255], #roads
[  0, 255 ,  0], #vegetation
[  0 ,255 ,255], #water
[153 ,  0 ,  0], #unspecified
[255 ,  0 ,0],#buidings 
[255 ,127 , 80]#cars
]    
counts = [0,0,0,0,0,0]
# [1944506, 3177633, 118664, 1327847, 917817, 132093]
print(labels.shape)
shaped = np.reshape(labels,(1920*3968,3)).tolist()
# print(np.unique(shaped, axis=0)    )
for i in shaped:
    for id in range(len(classes)):
        if i == classes[id]:
            counts[id] += 1
print(counts)
for count in counts:
    print(count/1920/3968*100)

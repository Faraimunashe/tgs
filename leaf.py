import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
#%matplotlib inline #uncomment if in notebook

def mask_leaf(im_name, external_mask=None):

    im = cv2.imread(im_name)
    im = cv2.blur(im, (5,5))

    height, width = im.shape[:2]

    mask = np.ones(im.shape[:2], dtype=np.uint8) * 2 #start all possible background
    '''
    #from docs:
    0 GC_BGD defines an obvious background pixels.
    1 GC_FGD defines an obvious foreground (object) pixel.
    2 GC_PR_BGD defines a possible background pixel.
    3 GC_PR_FGD defines a possible foreground pixel.
    '''

    #2 circles are "drawn" on mask. a smaller centered one I assume all pixels are definite foreground. a bigger circle, probably foreground.
    r = 100
    cv2.circle(mask, (int(width/2.), int(height/2.)), 2*r, 3, -3) #possible fg
    #next 2 are greens...dark and bright to increase the number of fg pixels.
    mask[(im[:,:,0] < 45) & (im[:,:,1] > 55) & (im[:,:,2] < 55)] = 1  #dark green
    mask[(im[:,:,0] < 190) & (im[:,:,1] > 190) & (im[:,:,2] < 200)] = 1  #bright green
    mask[(im[:,:,0] > 200) & (im[:,:,1] > 200) & (im[:,:,2] > 200) & (mask != 1)] = 0 #pretty white

    cv2.circle(mask, (int(width/2.), int(height/2.)), r, 1, -3) #fg

    #if you pass in an external mask derived from some other operation it is factored in here.
    if external_mask is not None:
        mask[external_mask == 1] = 1

    bgdmodel = np.zeros((1,65), np.float64)
    fgdmodel = np.zeros((1,65), np.float64)
    cv2.grabCut(im, mask, None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    #show mask
    plt.figure(figsize=(10,10))
    plt.imshow(mask)
    plt.show()

    #mask image
    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
    output = cv2.bitwise_and(im, im, mask=mask2)
    plt.figure(figsize=(10,10))
    plt.imshow(output)
    plt.show()

mask_leaf('coke.jpg', external_mask=None)
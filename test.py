#import the necessary packages
import numpy as np
import cv2

#read the image
image = cv2.imread('lf.jpg')

#convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#apply morphological operations
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

#find sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

#find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

#apply connected components
ret, markers = cv2.connectedComponents(sure_fg)

#add one to all labels so that sure background is not 0, but 1
markers = markers+1

#mark the unknown region with 0
markers[unknown==255] = 0

#apply watershed
markers = cv2.watershed(image, markers)
image[markers == -1] = [255,0,0]

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([26,97,100])
higher = np.array([255,255,255])
new_mask = cv2.inRange(hsv, lower, higher)

ccs, hrc = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(ccs) != 0:
    print("Length of contours =>",len(ccs))
    for c in ccs:
        if cv2.contourArea(c) > 500:
            print("Contour Area =>",cv2.contourArea(c))
            x,y,w,h = cv2.boundingRect(c)
            print(x,y,w,h)

#display the segmented image
cv2.imshow('segmented image', image)
cv2.imshow('tested image', new_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
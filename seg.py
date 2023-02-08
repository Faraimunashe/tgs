import cv2
import numpy as np

# Read the image
img = cv2.imread('leaf.jpg')

def grade1(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255,255,255), -1)

    # Apply the mask
    segmented = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
    lower = np.array([14,97,100])
    higher = np.array([255,255,255])
    new_mask = cv2.inRange(hsv, lower, higher)

    ccs, hrc = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ccs) != 0:
        print(len(ccs))
        for c in ccs:
            if cv2.contourArea(c) > 500:
                print(cv2.contourArea(c))
                # cv2.drawContours(new_mask)
                x,y,w,h = cv2.boundingRect(c)
                print(x,y,w,h)
                # cv2.rectangle(new_mask, (x,y), (x + w, y +h), (0,0,255), 2)
                #cv2.drawContours(new_mask, (x,y), w, (12,255,255), 3)


    # Display the segmented image
    cv2.imshow('Segmented Image', new_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#dark shade 2,
def grade2(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255,255,255), -1)

    # Apply the mask
    segmented = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
    lower = np.array([0,77,10])
    higher = np.array([255,255,255])
    new_mask = cv2.inRange(hsv, lower, higher)

    ccs, hrc = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ccs) != 0:
        print(len(ccs))
        for c in ccs:
            if cv2.contourArea(c) > 500:
                print(cv2.contourArea(c))
                # cv2.drawContours(new_mask)
                x,y,w,h = cv2.boundingRect(c)
                print(x,y,w,h)


    # Display the segmented image
    cv2.imshow('Segmented Image', new_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#a lighter dark 7,
def grade3(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255,255,255), -1)

    # Apply the mask
    segmented = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
    lower = np.array([7,97,10])
    higher = np.array([255,255,255])
    new_mask = cv2.inRange(hsv, lower, higher)

    ccs, hrc = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ccs) != 0:
        print(len(ccs))
        for c in ccs:
            if cv2.contourArea(c) > 500:
                print(cv2.contourArea(c))
                # cv2.drawContours(new_mask)
                x,y,w,h = cv2.boundingRect(c)
                print(x,y,w,h)


    # Display the segmented image
    cv2.imshow('Segmented Image', new_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# worst
def grade4(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255,255,255), -1)

    # Apply the mask
    segmented = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
    lower = np.array([0,97,0])
    higher = np.array([255,255,255])
    new_mask = cv2.inRange(hsv, lower, higher)

    ccs, hrc = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ccs) != 0:
        print(len(ccs))
        for c in ccs:
            if cv2.contourArea(c) > 500:
                print(cv2.contourArea(c))
                # cv2.drawContours(new_mask)
                x,y,w,h = cv2.boundingRect(c)
                print(x,y,w,h)


    # Display the segmented image
    cv2.imshow('Segmented Image', new_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



grade4(img=img)
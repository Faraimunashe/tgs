import cv2
import numpy as np

# Read the image
img = cv2.imread('leaf.jpg')

#Lemon Grade
def gradeLemon(path):
    # Read the image
    img = cv2.imread(path)
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
        return len(ccs)
        # for c in ccs:
        #     if cv2.contourArea(c) > 500:
        #         print(cv2.contourArea(c))
        #         # cv2.drawContours(new_mask)
        #         x,y,w,h = cv2.boundingRect(c)
        #         print(x,y,w,h)
        #         # cv2.rectangle(new_mask, (x,y), (x + w, y +h), (0,0,255), 2)
        #         #cv2.drawContours(new_mask, (x,y), w, (12,255,255), 3)


    # Display the segmented image
    # cv2.imshow('Segmented Image', new_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return 100


#dark shade 2,
def gradeRed(path):
    # Read the image
    img = cv2.imread(path)
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
        return len(ccs)
        # for c in ccs:
        #     if cv2.contourArea(c) > 500:
        #         print(cv2.contourArea(c))
        #         # cv2.drawContours(new_mask)
        #         x,y,w,h = cv2.boundingRect(c)
        #         print(x,y,w,h)


    # Display the segmented image
    # cv2.imshow('Segmented Image', new_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return 100

#a lighter dark 7,
def gradeOrange(path):
    # Read the image
    img = cv2.imread(path)
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
        return len(ccs)
        # for c in ccs:
        #     if cv2.contourArea(c) > 500:
        #         print(cv2.contourArea(c))
        #         # cv2.drawContours(new_mask)
        #         x,y,w,h = cv2.boundingRect(c)
        #         print(x,y,w,h)


    # Display the segmented image
    # cv2.imshow('Segmented Image', new_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return 100

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



#grade4(img=img)

def grader(path):
    if gradeLemon(path=path) <= 3:
        tbl = gradeLemon(path)
        return "X" + str(tbl) + "L"
    else:
        if gradeOrange(path=path) <= 3:
            tbf = gradeOrange(path)
            return "C" + str(tbf) + "F"

        else: 
            if gradeRed(path=path) <= 3:
                tbr = gradeRed(path)
                return "C" + str(tbr) + "R"
            
            return "X5R"

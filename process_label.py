from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import skimage
from skimage.feature import peak_local_max


def watershed(image):
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=1, labels=thresh)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = skimage.morphology.watershed(-D, markers, mask=thresh)

    for label in np.unique(labels):
        # If the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
    
        # Otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
    
        # Detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        # print(cnts)

        # for cnt in cnts:
        #     rect = cv2.minAreaRect(cnt)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     cv2.drawContours(image, [box], 0, (0, 0, 255), 1)

        c = max(cnts, key=cv2.contourArea)
    
        # Draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 1)
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    return image


if __name__ == '__main__':
    image = cv2.imread('debug.png')
    image = watershed(image)
    cv2.imwrite('thres_1.png', image)
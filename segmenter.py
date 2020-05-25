import cv2
import numpy as np

lowerBlue = np.array([100, 10, 10])
upperBlue = np.array([250, 200,200])

lowerBlack = np.array([0, 0, 0])
upperBlack = np.array([100, 100, 100])

compressionRatio = 200

def segmentary(image):
    blueLineMask = cv2.inRange(image, lowerBlue, upperBlue)
    image = cv2.blur(image,(5,5))
    blackLineMask = cv2.inRange(image, lowerBlack, upperBlack)
    
    return blueLineMask, blackLineMask

def readImageAndResize(filePath):
    image = cv2.imread(filePath)
    image = cv2.resize(image, (compressionRatio, compressionRatio))
    
    return image

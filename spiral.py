import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, atan
import cv2

filePath = input("image path: ")

#defining the center of image
x0, x1, y0, y1 = extent = [-17.8, 16, 15, -17]

#reading the image
image = cv2.imread(filePath)

#defining the ratio of compression
compressionRatio = 200
image = cv2.resize(image, (compressionRatio, compressionRatio))

#defining the interval of colors
lowerBlue = np.array([100, 10, 10])
upperBlue = np.array([250, 140,140])

lowerWhite = np.array([0, 0, 0])
upperWhite = np.array([255, 255, 255])

#segmentation of the image
blueLineMask = cv2.inRange(image, lowerBlue, upperBlue)
whiteLineMask = cv2.inRange(image, lowerWhite, upperWhite)

radius = []
thetas = []

#thinning the image
blueLineMask = cv2.ximgproc.thinning(blueLineMask)

#transform cartesian coordinates to polar
for i in range(len(blueLineMask)):
    for j in range(len(blueLineMask[0])):
        if(blueLineMask[i][j]):
            x = x0 + j * (x1 - x0) / (blueLineMask.shape[1] - 1)
            y = y1 + i * (y0 - y1) / (blueLineMask.shape[0] - 1)
            
            r = sqrt((x*x)+(y*y)) #finds the radius
            theta = np.arctan2(x, y) #finds the angle of the point

            #find the best K
            k = np.round((r - theta) / (2 * np.pi))

            radius.append(r)
            thetas.append(theta+2*k*np.pi)


#shows the graph angle vs radius
plt.plot(thetas, radius, '.')
plt.show()


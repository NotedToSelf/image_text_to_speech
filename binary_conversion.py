#Conversion to binary text image according to "Text Detection in Indoor/Outdoor Scenes" - Gatos, Pratikakais, Kepene, Perantonis

import cv2 as cv
import numpy as np
from skimage import restoration

In = cv.imread(filename='./Input.jpg', flags=cv.IMREAD_COLOR).astype(np.float32) / 255.0

#low pass weiner filter
img = cv.cvtColor(In, cv.COLOR_BGR2GRAY)
conv = np.ones((5, 5)) / 25
img = cv.filter2D(img, -1, conv)
img += 0.1 * img.std() * np.random.standard_normal(img.shape)
img = restoration.wiener(img, conv, 1100) 
cv.imwrite(filename='./Output.jpg', img=(img * 255.0).clip(0.0, 255.0).astype(np.uint8))


#Sauvola's approach for adaptive thresholding using k = 0.2






#Erik Bogeberg
#Conversion to binary text image 

import cv2
import numpy as np
import skimage.restoration

#original image
image = cv2.imread('./Input.jpg', 0) 

#remove background texture
#image = cv2.blur(image, (8,8))

#inverted image
invert_gray = cv2.bitwise_not(image)


#Adaptive Thresholding on both images 
_, image = cv2.threshold(image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, invert_gray = cv2.threshold(invert_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#pick the correct binary image
sum_orig = 0
sum_invert = 0
rows, cols = image.shape
for i in range(rows):
	for j in range(cols):
		if (image[i][j] > 230):
			sum_orig += 1
		if (invert_gray[i][j] > 230):
			sum_invert += 1 		

if (sum_orig < sum_invert):
	image = invert_gray


contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
	peri = cv2.arcLength(contours[i].astype(np.float32), True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

#Erode and Dialate, or find some way to remove remaining non character objects


#Text region detection


cv2.imwrite(filename='./Output.jpg', img=image)

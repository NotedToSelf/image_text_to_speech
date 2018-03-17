#Erik Bogeberg
#Conversion to binary text image 

import cv2
import numpy as np
import skimage.restoration

#original image
image = cv2.imread('./Input.png', 0) 

#remove background texture
#image = cv2.blur(image, (8,8))

#inverted image
invert_gray = cv2.bitwise_not(image)


#Adaptive Thresholding on both images 
_, image = cv2.threshold(image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, invert_gray = cv2.threshold(invert_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#pick the correct binary image
#I just sum up the white pixels below a threshold and use the greater
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


_, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Remove the background as a contur to preserve averages:
contours = contours[1:]

#Find the average height and width of contours
max_w = 0
max_h = 0
for cont in contours:
	x,y,w,h = cv2.boundingRect(cont)
	if w > max_w:
		max_w = w
	if h > max_h:
		max_h = h


#Crop out the valid contours and write them
char_count = 0
for cont in contours:
	x,y,w,h = cv2.boundingRect(cont)
	if w >= max_w - 40 and w <= max_w + 40 and h >= max_h - 20 and h <= max_h + 20:
		char_count+=1
		output = image[y:y+h, x:x+w]
		cv2.imwrite(str(char_count) + '.png', output)

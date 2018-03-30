#Erik Bogeberg
#Conversion to binary text image 

import cv2
import numpy as np
import skimage.restoration
import os

#clear data folder
cwd = os.getcwd()
dst = str(cwd) + '/data/images/'
for files in os.listdir(dst):
	if '.png' in files:
		os.remove(str(dst) + str(files))

#Read Source Image in grayscale
image = cv2.imread('./Input.png', 0) 
rows,cols = image.shape
area = rows * cols

#Denoise
#Filter strength 15 will remove noise as well as texture
image = cv2.fastNlMeansDenoising(image, None, 15, 13)

#inverted image
invert_gray = cv2.bitwise_not(image)

#Adaptive Thresholding on both images 
_, image = cv2.threshold(image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, invert_gray = cv2.threshold(invert_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Binary Image Selection
sum_orig = 0
sum_invert = 0
for i in range(rows):
	for j in range(cols):
		if (image[i][j] > 230):
			sum_orig += 1
		if (invert_gray[i][j] > 230):
			sum_invert += 1 		
if (sum_orig < sum_invert):
	image = invert_gray



#Obtain list of connected components
im, conts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Remove the background as a component:
conts = conts[1:]

#Find the max height and width of contours
max_w = 0
max_h = 0
for cont in conts:
	x,y,w,h = cv2.boundingRect(cont)
	if w > max_w:
		max_w = w
	if h > max_h:
		max_h = h


#Sort contours
#Source Credit: https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours
def precedence(contour, cols):
	tolerance = 20
	origin = cv2.boundingRect(contour)
	return ((origin[1] // tolerance) * tolerance) * cols+ origin[0]
conts.sort(key=lambda x:precedence(x, im.shape[1]))

#Find a close multiple of 28 to the size of found characters for padding
max_size = int((max_h / 28)+1) * 28

#Width and height tolerance of characters relative to max char size
w_tolerance = max_w // 1.2 
h_tolerance = max_h // 2

#Crop out the valid contours and write them
char_count = 0
for cont in conts:
	x,y,w,h = cv2.boundingRect(cont)
	if w >= max_w - w_tolerance and w <= max_w + w_tolerance and h >= max_h - h_tolerance and h <= max_h + h_tolerance:
		char_count+=1
		output = image[y:y+h, x:x+w]
		output = cv2.bitwise_not(output)
		r,c = output.shape
		#pad the image with black border
		horiz = int((max_size - r) / 2)
		vert = int((max_size - c) / 2)
		output = cv2.copyMakeBorder(output, horiz, horiz, vert, vert, cv2.BORDER_CONSTANT, value=[0])
		#resize to desired shape
		output = cv2.resize(output, (28, 28))
		cv2.imwrite('./data/images/' + str(char_count) + '.png', output)

# Handwritten digits recognition
# Support Vector Machine method
# HYPJUDY 2017.7.10
# http://hypjudy.github.io/

import cv2
import numpy as np
import os
import os.path
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib

# global parameters
READ_FOLDER = 'data567' # put image 5&6&7 into folder data567
WRITE_FOLDER = 'digit567'
PROCESS_FOLDER = 'process'
MODEL_PATH = 'models/mnist_svm_model_full.pkl'
THRESHOLD = 120 # discriminate digits from background
CROP_LEN = 10 # get rid of original image's black borders
RESOLUTION = 28 # 28*28 for mnist dataset

if not os.path.exists(WRITE_FOLDER):
    os.makedirs(WRITE_FOLDER)
if not os.path.exists(PROCESS_FOLDER):
    os.makedirs(PROCESS_FOLDER)

# Choose the corresponding labels
# 567
labels = np.array(
  [0,7,5,5,
  1,2,3,0,7,4,6,8,
  5,2,3,0,6,
  6,4,1,9,3,7,9,6, # 
  1,3,7,1,4,5,
  5,0,4,
  2,0,1,7,0,7,0,8,
  1,4,3,2,0,5,2,7, #
  1,8,8,1,9,2,5,0,0,
  5,1,8,0,6,8,
  2,6,6,7,8,6,6,4],
  np.uint8)

# 4
# labels = np.array(
#   [1,3,5,8,2,0,0,7,0,3,7,
#   5,1,0,0,0,6,
#   0,2,0,8,7,8,3,6,7,6,1,
#   1,3,9,2,4,5,7,9,6,9,3],
#   np.uint8)

# 3
# labels = np.array(
#   [0,6,2,8,0,8,8,7,1,3, # 
#   8,2,7,2,8,3,0,0,5,0,
#   2,0,3,8,0,7,0,8,7,0,
#   0,8,3,1,0,1,1,1,8,1,
#   3,8,0,8,8,1,5,3,6,9,
#   0,7,3,6,8,5,0,0,7,4],
#   np.uint8)

# 1
# labels = np.array(
#   [1,9,6,7,4,8,9,9, # 
#   3,5,4,7,0,8,7,4,5],
#   np.uint8)

# 0
# labels = np.array(
#   [1,2,4,7,6,7,3, # 
#   8,9,5,2,4,8,1],
#   np.uint8)
    
####################  kernels for dilating  ######################
# Used for connect broken digit
kernel_connect = np.array([[1,1,1], [1,0,1], [1,1,1]], np.uint8)
# Elliptical Kernel
kernel_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
# Cross-shaped Kernel
# to manipulate the orientation of dilution, large x means 
# horizonatally dilating more, large y means vertically dilating more
kernel_cross_h = cv2.getStructuringElement(cv2.MORPH_CROSS,(1, 8))
kernel_cross_w = cv2.getStructuringElement(cv2.MORPH_CROSS,(8, 1))


def is_vertical_writing(img):
	h, w = img.shape
	h_bin = np.zeros(h, np.uint16) # 0 to 65535
	w_bin = np.zeros(w, np.uint16)
	x, y = np.where(img == 255) # white
	for i in x:
		h_bin[i] = h_bin[i] + 1
	for j in y:
		w_bin[j] = w_bin[j] + 1
	# calculate the number of continuous zero (background)
	# areas in vertical (h) and horizontal (w) orientation
	n_h_zero_area = 0
	for i in range(h - 1):
		if h_bin[i] == 0 and h_bin[i + 1] != 0:
			n_h_zero_area = n_h_zero_area + 1
	n_w_zero_area = 0
	for i in range(w - 1):
		if w_bin[i] == 0 and w_bin[i + 1] != 0:
			n_w_zero_area = n_w_zero_area + 1

	if n_h_zero_area > n_w_zero_area: # sparse vertically
		return True # dense horizontally
	return False


# Given a string of digits, return each digit
def split_digits_str(s, prefix_name, is_vertical):
	# to read digits of a string in order, rotate the image
	# and let the leading digit lying in the bottom
	# since cv2.findContours from bottom to top
	if is_vertical:
		s = np.rot90(s, 2)
	else:
		s = np.rot90(s)
	
	# if each digit is continuous (not broken), needn't dilate
	# so for image 5/6/7, use (*); otherwise, use (**)
	# for image 0/1, iter=2; for image 2/3/4, iter=1
	# s_copy = s.copy() # (*)
	s_copy = cv2.dilate(s, kernel_connect, iterations = 1) # (**)
	
	s_copy2 = s_copy.copy()
	contours, hierarchy = cv2.findContours(s_copy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	idx = 0
	digits_arr = np.array([])
	for contour in contours:
		idx = idx + 1
		[x, y, w, h] = cv2.boundingRect(contour)
		digit = s_copy[y:y + h, x:x + w]

		# in order to keep the original scale of digit
		# pad rectangles to squares before resizing
		pad_len = (h - w) / 2
		if pad_len > 0: # to pad width
			# Forms a border around an image: top, bottom, left, right
			digit = cv2.copyMakeBorder(digit, 0, 0, pad_len, pad_len, cv2.BORDER_CONSTANT, value=0)
		elif pad_len < 0: # to pad height
			digit = cv2.copyMakeBorder(digit, -pad_len, -pad_len, 0, 0, cv2.BORDER_CONSTANT, value=0)
		pad = digit.shape[0] / 5 # avoid the digit directly connect with border, leave around 4 pixels
		digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
		digit = cv2.resize(digit, (RESOLUTION, RESOLUTION), interpolation = cv2.INTER_AREA)
		
		digit = np.rot90(digit, 3) # rotate back to horizontal orientation
		digit_name = os.path.join(WRITE_FOLDER, prefix_name + '_n' + str(idx) + '.png')
		cv2.imwrite(digit_name, digit)

		# a digit: transform 2D array to 1D array
		digit = np.concatenate([(digit[i]) for i in range(RESOLUTION)])
		digits_arr = np.append(digits_arr, digit)
	
	# transform 1D array to 2D array
	digits_arr = digits_arr.reshape((digits_arr.shape[0] / (RESOLUTION * RESOLUTION), -1))
	return digits_arr

def load_digits_arr_from_folder():
	digits_arr = np.array([])
	for filename in os.listdir(WRITE_FOLDER):
		img = cv2.imread(os.path.join(WRITE_FOLDER, filename), 0)
		fn = os.path.splitext(filename)[0] # without extension
		if img is not None:
			digit = np.concatenate([(img[i]) for i in range(RESOLUTION)])
			digits_arr = np.append(digits_arr, digit)
	digits_arr = digits_arr.reshape((-1, RESOLUTION * RESOLUTION))
	return digits_arr


def find_digits_str(folder):
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename), 0)
		fn = os.path.splitext(filename)[0] # without extension
		if img is not None:
			height, width = img.shape
			crop_name = os.path.join(PROCESS_FOLDER, fn + '_crop.png')
			thre_name = os.path.join(PROCESS_FOLDER, fn + '_thre.png')
			dil_name = os.path.join(PROCESS_FOLDER, fn + '_dil.png')
			cropped = img[CROP_LEN:height - CROP_LEN, CROP_LEN:width - CROP_LEN]
			ret, thresh_img = cv2.threshold(cropped, THRESHOLD, 255, cv2.THRESH_BINARY_INV) # INV for black text
			cv2.imwrite('7_A4_threshold_140.png', thresh_img)
			is_vertical = is_vertical_writing(thresh_img)

			dilated = cv2.dilate(thresh_img, kernel_ellip, iterations = 1)
			if is_vertical:
				dilated = cv2.dilate(dilated, kernel_cross_h, iterations = 10)
			else:
				dilated = cv2.dilate(dilated, kernel_cross_w, iterations = 10)
			
			contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			idx = 0
			for contour in contours:
				[x, y, w, h] = cv2.boundingRect(contour)
				if is_vertical and (w < 30 or w > 100 or h < 70 or h > 520):
					continue
				elif (is_vertical == False) and (h < 30 or h > 100 or w < 70 or w > 520):
					continue

				idx = idx + 1
				digits_str = thresh_img[y:y + h, x:x + w]
				save = np.rot90(digits_str)
				cv2.imwrite(str(idx)+'str.png', save)
				# digits_arr = split_digits_str(digits_str, fn + '_s' + str(idx), is_vertical)

				cv2.rectangle(thresh_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
			
			# cv2.imwrite(crop_name, cropped)
			cv2.imwrite(thre_name, thresh_img)
			# cv2.imwrite(dil_name, dilated)

def train_mnist_svm():
	if os.path.isfile(MODEL_PATH):
		classifier = joblib.load(MODEL_PATH)
	else:
		mnist = fetch_mldata('MNIST original', data_home='./')
		X_data = mnist.data / 255.0
		Y = mnist.target
		classifier = svm.SVC(C=5,gamma=0.05)
		classifier.fit(X_data, Y)
		joblib.dump(classifier, MODEL_PATH)
	return classifier

def predict_mnist_svm(digits_arr):
	classifier = train_mnist_svm()
	digits_arr = digits_arr / 255.0
	predicted = classifier.predict(digits_arr)
	# labels are defined at the begining
	print ('labels', labels)
	print ('predicted', predicted)
	boolarr = (labels == predicted)
	print (boolarr)
	correct = np.sum(boolarr)
	num = boolarr.shape[0]
	acc = correct * 1.0 / num
	print('test accuracy of %s is %d / %d = %f' % (WRITE_FOLDER, correct, num, acc))


find_digits_str(READ_FOLDER)
digits_arr = load_digits_arr_from_folder()
predict_mnist_svm(digits_arr)


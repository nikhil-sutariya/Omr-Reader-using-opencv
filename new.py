import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours

ANSWER_KEY = {
    0: 2, 1: 0, 2: 3, 3: 2, 4: 1, 5: 1, 6: 0, 7: 3, 8: 0, 9: 1,
    10: 3, 11: 2, 12: 3, 13: 0, 14: 0, 15: 0, 16: 0, 17: 3, 18: 2, 19: 2, 
    20: 2, 21: 2, 22: 2, 23: 1, 24: 2, 25: 3, 26: 2, 27: 2, 28: 0, 29: 2 
}

# Step - 1 => Open the image, convert it in a grayscale 

image = cv2.imread("test.png")
bigger = imutils.resize(image, height=750)
gray = cv2.cvtColor(bigger, cv2.COLOR_BGR2GRAY)

# Step - 2 => Detect the edges of gray image (It detect boundareis of image)
# first min_threshold and max_threshold
image_mean = bigger[:,:,0]
mean = image_mean.mean()

min_threshold = 0.66 * mean
max_threshold = 1.33 * mean

# print("Min -> ", min_threshold, '\n', 'Max -> ', max_threshold)

# edged = cv2.Canny(gray, min_threshold, max_threshold)

# Step - 3 => Thresholding - convert images to binary images
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Step - 4 => Contouring - finding all white object from black background
# paper = four_point_transform(image, docCnt.reshape(4, 2))
# warped = four_point_transform(gray, docCnt.reshape(4, 2))

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# cv2.drawContours(bigger, cnts, -1, (0, 255, 0), 2)

# questionCnts = []

# for c in cnts:
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	ar = w / float(h)

# 	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
# 		questionCnts.append(c)

# questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
# correct = 0

# for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
#     cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
#     bubbled = None

#     for (j, c) in enumerate(cnts): 
#         mask = np.zeros(thresh.shape, dtype="uint8")
#         cv2.drawContours(mask, [c], -1, 255, -1)
#         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#         total = cv2.countNonZero(mask)
    
#         if bubbled is None or total > bubbled[0]:
#             bubbled = (total, j)

#     color = (0, 0, 255)
#     k = ANSWER_KEY[q]

#     if k == bubbled[1]:
#         color = (0, 255, 0)
#         correct += 1

#     cv2.drawContours(bigger, [cnts[k]], -1, color, 3)
# print("[Correct] - ", correct)

# cv2.imshow("Bigger", bigger)
# cv2.waitKey(0)
# cv2.imshow("Gray", gray)
# cv2.waitKey(0)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.imshow("Thresh", thresh)
# cv2.waitKey(0)
# cv2.imshow("Paper", paper)
# cv2.waitKey(0)
# cv2.imshow("Warped", warped)
# cv2.waitKey(0)
cv2.imshow("Answer", bigger)
cv2.waitKey(0)
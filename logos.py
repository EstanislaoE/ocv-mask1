
#import numpy as np 
import cv2 
image_read = cv2.imread("mission.jpg", cv2.IMREAD_GRAYSCALE)
# print(image_read)  [[]]
cv2.imwrite("Mizz.png", image_read)


# image_read = cv2.imread("mission.jpg", cv2.IMREAD_COLOR)
# b, g, r = cv2.split(image_read)
# cv2.imwrite("b_image.png", b )
# cv2.imwrite("g_image.png", g)
# cv2.imwrite("r_image.png", r)
# cv2.imwrite("merged_bgr_image.png", cv2.merge((b, g, r)))
# import cv2 
# import numpy as np 
# import matplotlib.pyplot as plt 

# SCALE = 3

# PAPER_W = 210 * SCALE 
# PAPER_H = 297 * SCALE 

# def load_image(path, scale = 0.7):
#     img = cv2.imread(path)
#     if img is None: 
#         raise FileNotFoundError(f"Could not laod image {path}")
#     #img_resized = cv2.resize(img, (0 , 0), None, scale, scale)
#     #return img_resized
#     return img 

# def show_image(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(6, 8))
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img)
#     plt.show()

# img_original = load_image(path="meas.jpg")
# print("Image shape: ", img_original.shape)


# def preprocess_image(img, thresh_1 = 50, thresh_2 = 150):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)

#     #Use CLAHE for better contrast 
#     clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
#     img_clahe = clahe.apply(img_blur)


#     edges = cv2.Canny(img_clahe, thresh_1, thresh_2)
#     _, thresh = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



#     combined = cv2.bitwise_or(edges, thresh)

#     kernel = np.ones((5, 5), np.uint8)
#     dilated = cv2.dilate(combined, kernel, iterations = 2)
#     closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations= 3 )

    
#     # img_preprocessed = img_closed.copy()

#     img_each_step = {'img_gray' : img_gray,
#                      'img_blur': img_blur,
#                     #  'img_canny': img_canny,
#                     #  'img_thresh': img_thresh,
#                     #  'img_combined': img_combined,
#                     # 'img_dilated': img_dilated
#                      }
    
#     return closed, img_each_step 
#     #return img_closed

# #img_original = load_image("meas.jpg")

# img_preprocessed, img_each_step = preprocess_image(img_original)
# show_image(img_preprocessed)

# show_image(img_each_step['img_gray'])
# show_image(img_each_step['img_blur'])
# # show_image(img_each_step['img_canny'])
# # show_image(img_each_step['img_dilated'])

# #show_image(img_each_step['img_thresh'])
# #show_image(img_each_step['img_canny'])
# show_image(img_preprocessed)

# print("Test A Here")


# def find_contours(img_preprocessed, img_original, epsilon_param=0.01, min_area= 5000 ):
#     contours, _ = cv2.findContours(img_preprocessed,
#                                             cv2.RETR_EXTERNAL,
#                                            cv2.CHAIN_APPROX_SIMPLE)
#     img_contour = img_original.copy()
#     #cv2.drawContours(img_contour, contours, -1, (203, 192, 255), 6)
#     polygons = []

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area < min_area: 
#             continue
#         # epsilon = epsilon_param * cv2.arcLength(curve = contour, closed = True)
#         # polygon = cv2.approxPolyDP(curve = contour, epsilon = epsilon,  closed = True)
#         # polygon = polygon.reshape(-1, 2)
#         # polygons.append(polygon)

#         perimeter = cv2.arcLength(contour, True)
#         epsilon = epsilon_param * perimeter
#         polygon = cv2.approxPolyDP(contour, epsilon, True)

#         # for point in polygon: 
#         #     img_contour = cv2.circle(img = img_contour, center = point, radius = 8, color = (0 , 240, 0), thickness = 1)
#         if 4 <= len(polygon) <= 6:
#             if len(polygon) > 4: 
#                 new_epsilon = 0.1*perimeter
#                 polygon = cv2.approxPolyDP(contour, new_epsilon, True)
#                 polygon = polygon.reshape(-1, 2)[:4]
            
#             polygon = polygon.reshape(-1, 2)
#             polygon.append(polygon)

#             cv2.drawContours(img_contour, [polygon.astype(int)], 
#                              -1, (203, 192, 255), 3)
#             for point in polygon: 
#                 cv2.circle(img_contour, tuple(point.astype(int)), 8, (0, 240, 0), -1)

#     return polygon, img_contour 

# polygons, img_contours = find_contours(img_preprocessed, img_original)
# print("img_contours: ")
# show_image(img_contours)

# # print("Polygons[0]: ")
# # print(polygons[0])
# # print(" ")

# #Codeblock 9 
# def reorder_coords(polygon):
#     polygon = np.array(polygon).reshape(-1, 2)

#     if len(polygon) < 4:
#         raise ValueError(f"Expected 4 points, got  {len(polygon)}")
#         h, w = img_original.shape[:2]
#         return np.float32([[0, 0], [w,0], [0,h], [w,h]])
    
#     hull = cv2.convexHull(polygon.reshape(-1, 1, 2)).reshape(-1, 2)
    
#     # if len(polygon) > 4: 
#     #     polygon = polygon[:4]
 
#     # if polygon.shape != (4, 2): 
#     #     polygon = polygon.reshape( -1, 2[:4])

#     rect_coords = np.zeros((4, 2))

#     center = np.mean(hull, axis = 0)
#     angles = np.arctan2(hull[:,1]-center[1], hull[:0]-center[0])
#     hull = hull[np.argsort(angles)]


#     rect_coords[0] = hull[0]
#     rect_coords[1] = polygon[np.argmax(sums)]

#     #difference of x-y coordinates 
#     diffs = polygon[:,0] - polygon[:,1]
#     rect_coords[1] = polygon[np.argmin(diffs)]
#     rect_coords[2] = polygon[np.argmax(diffs)]

#     return rect_coords 


# # img_orginal = load_image("meas.jpg")
# # img_preprocessed = preprocess_image(img_original)


# #codeblock 10 
# # rect_coords = np.float32(reorder_coords(polygons[0]))
# # print(" rect_coords: ")
# # print(rect_coords)
# # polygons, img_contours = find_contours(img_preprocessed, img_original, epsilon_param= 0.04)
# # show_image(img_contours)


# #check if we found any polygons 
# if len(polygons) > 0:
#     print(f"Found {len(polygons)} potential quadrilaterals")
    
#     for i, polygon in enumerate(polygons):
#         try:
#             rect_coords = np.float32(reorder_coords(polygon))
#             print(f"\nRectangle {i+1} coordinates:")
#             print(rect_coords)
            
#             # You can proceed with warping here if coordinates look correct
#         except Exception as e:
#             print(f"\nError processing polygon {i+1}: {e}")
#             print("Polygon points:", polygon)
# else:
#     print("\nNo quadrilaterals found. Try:")
#     print("1. Adjusting epsilon_param in find_contours()")
#     print("2. Changing threshold values in preprocess_image()")
#     print("3. Checking your input image for clear rectangular objects")




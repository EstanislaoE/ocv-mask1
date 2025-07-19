import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

SCALE = 3

PAPER_W = 210 * SCALE 
PAPER_H = 297 * SCALE 

def load_image(path, scale = 0.7):
    img = cv2.imread(path)
    if img is None: 
        raise FileNotFoundError(f"Could not laod image {path}")
    #img_resized = cv2.resize(img, (0 , 0), None, scale, scale)
    #return img_resized
    return img 

def show_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 8))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()

img_original = load_image(path="meas.jpg")
# show_image(img_original)
print(img_original.shape)

def preprocess_image(img, thresh_1 = 57, thresh_2=232):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, thresh_1, thresh_2)

    kernel = np.ones((3, 3))
    img_dilated = cv2.dilate(img_canny, kernel, iterations = 1)
    img_closed = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE, kernel, iterations= 4 )
    
    img_preprocessed = img_closed.copy()

    img_each_step = {'img_dilated': img_dilated,
                     'img_canny': img_canny,
                     'img_blur': img_blur,
                     'img_gray' : img_gray}
    
    return img_preprocessed, img_each_step 

img_preprocessed, img_each_step = preprocess_image(img_original)
# show_image(img_each_step['img_gray'])
# show_image(img_each_step['img_blur'])
# show_image(img_each_step['img_canny'])
# show_image(img_each_step['img_dilated'])
show_image(img_preprocessed)
print("Test A Here")


def find_contours(img_preprocessed, img_original, epsilon_param=0.04 ):
    contours, hierarchy = cv2.findContours(image = img_preprocessed,
                                           mode = cv2.RETR_EXTERNAL,
                                           method = cv2.CHAIN_APPROX_NONE)
    img_contour = img_original.copy()
    cv2.drawContours(img_contour, contours, -1, (203, 192, 255), 6)

    polygons = []
    for contour in contours:
        epsilon = epsilon_param * cv2.arcLength(curve = contour, closed = True)
        polygon = cv2.approxPolyDP(curve = contour, epsilon = epsilon,  closed = True)
        polygon = polygon.reshape(-1, 2)
        polygons.append(polygon)

        for point in polygon: 
            img_contour = cv2.circle(img = img_contour, center = point, radius = 8, color = (0 , 240, 0), thickness = 1)
        
    return polygon, img_contour 

polygons, img_contours = find_contours(img_preprocessed, img_original, epsilon_param = 0.04)


show_image(img_contours)
print("Polygons[0]: ")
print(polygons[0])
print(" ")


# #Codeblock 9 
# def reorder_coords(polygon):
#     polygon = np.array(polygon).reshape(-1, 2)

#     if len(polygon) > 4: 
#         polygon = polygon[:4]
 
#     # if polygon.shape != (4, 2): 
#     #     polygon = polygon.reshape( -1, 2[:4])

#     rect_coords = np.zeros((4, 2))

#     add = polygon.sum(axis = 1)
#     rect_coords[0] = polygon[np.argmin(add)]
#     rect_coords[3] = polygon[np.argmax(add)]

#     #difference of x-y coordinates 
#     subtract = polygon[:0] - polygon[:,1]
#     rect_coords[1] = polygon[np.argmin(subtract)]
#     rect_coords[2] = polygon[np.argmax(subtract)]

#     return rect_coords 

# #codeblock 10 
# rect_coords = np.float32(reorder_coords(polygons[0]))
# print(" rect_coords: ")
# print(rect_coords)









#  import cv2 
# import numpy as np 
# #Load the image 
# img = cv2.imread('meas.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# PIXELS_PER_INCH = 100

# ret, thresh = cv2.threshold(
#     gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )

# for cnt in contours: 
#     # area = cv2.contourArea(cnt)

#     # x, y, w, h = cv2.boundingRect(cnt)
#     # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     # cv2.putText(img, str(area), (x, y), 
#     #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     pixel_area = cv2.contourArea(cnt)

#     #convert
#     area_inches =  pixel_area / (PIXELS_PER_INCH ** 2)

#     x, y, w, h = cv2.boundingRect(cnt)

#     w_inches = w / PIXELS_PER_INCH
#     h_inches = h / PIXELS_PER_INCH

#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     text = f"{area_inches: 0.2f} in2 ({w_inches: 0.2f} x {h_inches: 0.2f} in)"
#     cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# cv2.imshow('MEASUREMENT.jpg', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

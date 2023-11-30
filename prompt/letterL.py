import cv2
import numpy as np
# letter L
def generate_pick_probability_map(img):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    
    # Thresholding the image
    # personally I change 127 to 100
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order, and keep the largest one
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Create a blank probability map
    prob_map = np.zeros_like(img, dtype=float)
    
    # Traverse each pixel in the image
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # For each pixel, calculate its distance to the contour
            dist = cv2.pointPolygonTest(contour, (x, y), True)
            print(dist)
            
            # If the pixel is inside the contour, update the probability map
            if dist > 0:
                prob_map[y, x] = dist
    
    # Normalize the probability map
    prob_map /= np.max(prob_map)
    
    return prob_map
import cv2
import numpy as np


def pick_probability_map(image_path):
    # Load the image
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (28,28))
    # Binarize the image
    _, binarized = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    
    # Create an empty distance map
    distance_map = np.zeros_like(image, dtype=float)
    
    # Calculate distance to the contour for each pixel
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            distance_map[y, x] = cv2.pointPolygonTest(contour, (x, y), True)
    
    # Make negative distances zero
    distance_map[distance_map < 0] = 0
    
    # Normalize to get probabilities
    probability_map = distance_map / distance_map.sum()
    
    return probability_map




# Example usage:

### self defined code

image_path = 'prompt/img/L.png'
# 
# image_path = 'prompt/img/V.png'
image_path = 'prompt/img/O.png'
prob_map = generate_pick_probability_map(image_path)
# prob_map = pick_probability_map(image_path)


import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(prob_map)
# remove the axis
plt.axis('off')
plt.subplot(1,2,2)

img = cv2.imread(image_path)
img = cv2.resize(img, (28,28))
# _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
plt.axis('off')
# remove the axis
plt.imshow(img)
plt.show()
import cv2
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
# distance
# def calculate_placement_probability(picked_img, placed_img, threshold=100):
#     # Convert images to grayscale
#     picked_gray = cv2.cvtColor(picked_img, cv2.COLOR_BGR2GRAY)
#     placed_gray = cv2.cvtColor(placed_img, cv2.COLOR_BGR2GRAY)
    
#     # Threshold the images
#     _, picked_thresh = cv2.threshold(picked_gray, threshold, 255, cv2.THRESH_BINARY)
#     _, placed_thresh = cv2.threshold(placed_gray, threshold, 255, cv2.THRESH_BINARY)
    
#     # Find contours
#     picked_contours, _ = cv2.findContours(picked_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     placed_contours, _ = cv2.findContours(placed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Select contours with maximum area
#     picked_contour = max(picked_contours, key=cv2.contourArea)
#     placed_contour = max(placed_contours, key=cv2.contourArea)
    
#     # Find the center of mass of the contours
#     picked_moments = cv2.moments(picked_contour)
#     picked_cx = int(picked_moments['m10'] / picked_moments['m00'])
#     picked_cy = int(picked_moments['m01'] / picked_moments['m00'])
    
#     placed_moments = cv2.moments(placed_contour)
#     placed_cx = int(placed_moments['m10'] / placed_moments['m00'])
#     placed_cy = int(placed_moments['m01'] / placed_moments['m00'])
#     print(picked_cx,picked_cy, placed_cx, placed_cy)
#     # Create a probability map
#     probability_map = np.zeros_like(placed_gray, dtype=np.float32)
#     for y in range(probability_map.shape[0]):
#         for x in range(probability_map.shape[1]):
#             distance = np.sqrt((x - placed_cx - (14 - picked_cx))**2 + (y - placed_cy- (14 - picked_cy))**2)
#             probability_map[y, x] = np.exp(-distance/3)
            
#     # Normalize the probability map
#     probability_map /= np.sum(probability_map)
    
#     return probability_map
# alignment
def calculate_placement_probability(picked_img, placed_img, threshold=100):
    # Preprocessing: Convert images to grayscale and threshold
    picked_gray = cv2.cvtColor(picked_img, cv2.COLOR_BGR2GRAY)
    _, picked_bin = cv2.threshold(picked_gray, threshold, 255, cv2.THRESH_BINARY)
    
    placed_gray = cv2.cvtColor(placed_img, cv2.COLOR_BGR2GRAY)
    _, placed_bin = cv2.threshold(placed_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Generate probability map
    prob_map = np.zeros_like(placed_gray, dtype=float)
    for x in range(placed_bin.shape[1]):
        for y in range(placed_bin.shape[0]):
            translation = np.array([x, y]) - np.array([placed_bin.shape[1]/2,placed_bin.shape[1]/2])
            M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
            picked_translated = cv2.warpAffine(picked_bin, M, (placed_bin.shape[1], placed_bin.shape[0]))
            intersection = np.logical_and(picked_translated, placed_bin).sum()
            prob_map[y, x] = intersection
            
    # Normalize the probability map
    prob_map /= prob_map.sum()
    
    return prob_map
# Load images
letter_image_path = "prompt/img/picked.png"
letter_image_path = "prompt/img/L.png"
bowl_image_path = "prompt/img/placed.png"
letter_img = cv2.imread(letter_image_path)
# letter_img = cv2.flip(letter_img, 0)
bowl_img = cv2.imread(bowl_image_path)
letter_img = cv2.resize(letter_img, (28,28))
bowl_img = cv2.resize(bowl_img, (28,28))
prob_map = calculate_placement_probability(letter_img, bowl_img)




# prob_map = generate_place_position_map(letter_img, bowl_img)


plt.subplot(1,3,1)
plt.imshow(prob_map)
plt.subplot(1,3,2)
plt.imshow(letter_img)
plt.subplot(1,3,3)
plt.imshow(bowl_img)
plt.show()

import cv2

def place_position(picked_img_path, placed_img_path, threshold=100):
    # Open the images
    picked_img = cv2.imread(picked_img_path, cv2.IMREAD_GRAYSCALE)
    picked_img = cv2.resize(picked_img, (28,28))
    placed_img = cv2.imread(placed_img_path, cv2.IMREAD_GRAYSCALE)
    placed_img = cv2.resize(placed_img, (28,28))
    # Binary thresholding
    _, picked_binary = cv2.threshold(picked_img, threshold, 255, cv2.THRESH_BINARY)
    _, placed_binary = cv2.threshold(placed_img, threshold, 255, cv2.THRESH_BINARY)
    

    
    # Calculating the placement probability map
    probability_map = np.zeros_like(placed_binary, dtype=float)
    for i in range(28):
        for j in range(28):
            # Calculating the overlap between the picked and placed objects for each position
            overlap = placed_binary * np.roll(np.roll(picked_binary, i-14, axis=0), j-14, axis=1)
        
            probability_map[i, j] = overlap.sum()
    
    # Normalizing the probability map
    probability_map /= probability_map.sum()
    
    return probability_map

# Testing the function and displaying the probability map

# Testing the function and displaying the probability map
# probability_map = place_position(letter_image_path, bowl_image_path)

# plt.imshow(probability_map, cmap='hot')
# plt.colorbar()
# plt.title('Placement Probability Map')
# plt.show()

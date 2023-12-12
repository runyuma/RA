import cv2
import numpy as np

def generate_pick_probability_map(img, threshold=100,deterministic = False):
    # Load the image in grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding the image
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
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
            
            # If the pixel is inside the contour, update the probability map
            if dist > 0:
                if not deterministic:
                    prob_map[y, x] = np.clip(dist,0,3)
                else:
                    prob_map[y, x] = dist
                # prob_map[y, x] = dist
    
    # Normalize the probability map
    prob_map /= prob_map.sum()
    
    return prob_map

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
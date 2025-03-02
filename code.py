import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\mahi9\Downloads\elon.jpeg"
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or could not be loaded.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
resized = cv2.resize(image, (300, 300))

# Rotate the image 90 degrees clockwise
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Flip the image horizontally
flipped = cv2.flip(image, 1)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# Perform edge detection using Canny
edges = cv2.Canny(image, 50, 150)

# Apply binary thresholding
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Apply dilation
dilated = cv2.dilate(edges, kernel, iterations=1)

# Apply erosion
eroded = cv2.erode(edges, kernel, iterations=1)

# Compute morphological gradient
gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

# Apply Sobel edge detection (X and Y directions)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)

# Apply Laplacian edge detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Convert the image to different color spaces
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Perform histogram equalization
equalized = cv2.equalizeHist(gray)

# Draw a rectangle on the image
rectangle = image.copy()
cv2.rectangle(rectangle, (50, 50), (250, 250), (0, 255, 0), 3)

# Draw a circle on the image
circle = image.copy()
cv2.circle(circle, (150, 150), 50, (255, 0, 0), -1)

# Draw a line on the image
line = image.copy()
cv2.line(line, (50, 50), (250, 250), (0, 0, 255), 3)

# Store all processed images and their respective titles
images = [gray, resized, rotated, flipped, blurred, edges, thresholded, adaptive_thresh, dilated, eroded,
          gradient, sobelx, sobely, laplacian, hsv, lab, equalized, rectangle, circle, line]

titles = ["Grayscale", "Resized", "Rotated", "Flipped", "Blurred", "Edges", "Thresholded", "Adaptive Thresh",
          "Dilated", "Eroded", "Gradient", "Sobel X", "Sobel Y", "Laplacian", "HSV", "LAB", "Equalized",
          "Rectangle", "Circle", "Line"]

# Create a subplot grid of 5 rows and 4 columns
fig, axes = plt.subplots(5, 4, figsize=(15, 15))
axes = axes.ravel()

# Display each processed image with its title below it
for i in range(len(images)):
    axes[i].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)  # Convert grayscale images properly
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
    axes[i].set_xlabel(titles[i], fontsize=10, labelpad=5)  # Set the title below each image

# Adjust spacing to ensure text is correctly positioned below each image
plt.subplots_adjust(hspace=0.6, wspace=0.3)

# Display the final plot
plt.show()

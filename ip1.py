import cv2
from PIL import Image

# Read the image
image = cv2.imread(r"nature.jpg", 1)

# Get the dimensions of the image (height, width)
(h, w) = image.shape[:2]

# Calculate the center of the image
H = int(h / 2)
W = int(w / 2)

# Iterate through the image and change the pixel values
for i in range(h):
    for j in range(w):
        # If the pixel is at the center line or border
        if j == W or i == H or j == W - 1 or i == H - 1:
            image[i, j] = (0, 0, 0)  # Set the pixel to black

# Convert the image from BGR (OpenCV format) to RGB (PIL format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the numpy array image to a Pillow Image object
pil_image = Image.fromarray(image_rgb)

# Save the image as a PDF
pil_image.save("Mahesh.pdf", "PDF")

# Display the modified image using OpenCV
cv2.imshow('split', image)

# Wait for a key press and close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
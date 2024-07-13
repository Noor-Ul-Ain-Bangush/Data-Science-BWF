# <<<----------Image Tutorial------------>>>

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the image using PIL and convert it to a NumPy array
img = np.asarray(Image.open('E:/Data-Science-BWF-Noor Ul Ain (Task9)/pythonProject/stinkbug.png'))
print(repr(img))

# Display the original image
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')  # Hide the axes

# Convert the image to grayscale by extracting the luminance channel
lum_img = img[:, :, 0]

# Display the grayscale image
plt.subplot(2, 3, 2)
plt.title('Grayscale Image')
plt.imshow(lum_img, cmap='gray')
plt.axis('off')  # Hide the axes

# Display the grayscale image with a hot colormap
plt.subplot(2, 3, 3)
plt.title('Hot Colormap')
plt.imshow(lum_img, cmap='hot')
plt.axis('off')  # Hide the axes

# Display the grayscale image with a nipy_spectral colormap
plt.subplot(2, 3, 4)
plt.title('Nipy Spectral Colormap')
imgplot = plt.imshow(lum_img, cmap='nipy_spectral')
plt.axis('off')  # Hide the axes

# Display the grayscale image with a colorbar
plt.subplot(2, 3, 5)
plt.title('With Colorbar')
imgplot = plt.imshow(lum_img, cmap='nipy_spectral')
plt.colorbar()

# Adjust the contrast of the grayscale image
plt.subplot(2, 3, 6)
plt.title('Adjusted Contrast')
plt.imshow(lum_img, cmap='gray', clim=(0, 175))
plt.axis('off')  # Hide the axes

plt.tight_layout()
plt.show()

# Display histogram of grayscale values
plt.figure(figsize=(10, 4))
plt.title('Histogram of Grayscale Values')
plt.hist(lum_img.ravel(), bins=range(256), fc='k', ec='k')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
path = Path('/mnt/EncryptedDisk2/BreastData/Studies/CLAM/masks/744_a.jpg')
img = mpimg.imread(path)

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

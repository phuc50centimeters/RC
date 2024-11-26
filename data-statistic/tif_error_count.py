import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

file_path = '../INPUT_DATA/Hima/2019040118/AWS_20190401180000.tif'

# List invalid value
invalid_values = [-np.inf, -9999, np.nan]

# Read the tif file
with rasterio.open(file_path) as src:
    data = src.read(1)

# Create a canvas
fig, ax = plt.subplots(figsize=(15, 6))

# Draw a rect for each pixel
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        value = data[row, col]
        
        if value in invalid_values:
            color = 'white'
        elif value > 0:
            color = 'blue'
        elif value == 0:
            color = 'orange'
        
        rect = patches.Rectangle((col, row), 1, 1, facecolor=color, edgecolor='none')
        ax.add_patch(rect)

ax.set_xlim(0, data.shape[1])
ax.set_ylim(0, data.shape[0])
ax.invert_yaxis() 

ax.axis('off')

plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
plt.close()


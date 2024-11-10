import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Đường dẫn đến tệp TIFF
file_path = '../INPUT_DATA/Hima/2019040118/AWS_20190401180000.tif'

# Giá trị không hợp lệ
invalid_values = [-np.inf, -9999, np.nan]

# Đọc tệp TIFF
with rasterio.open(file_path) as src:
    data = src.read(1)  # Đọc băng đầu tiên

# Tạo một hình vẽ
fig, ax = plt.subplots(figsize=(15, 6))

# Vẽ hình chữ nhật cho từng ô
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        value = data[row, col]
        
        # Chọn màu sắc dựa trên giá trị
        if value in invalid_values:
            color = 'white'  # Màu trắng cho giá trị không hợp lệ
        elif value > 0:
            color = 'blue'   # Màu xanh cho giá trị > 0
        elif value == 0:
            color = 'orange' # Màu cam cho giá trị = 0
        
        # Vẽ hình chữ nhật
        rect = patches.Rectangle((col, row), 1, 1, facecolor=color, edgecolor='none')
        ax.add_patch(rect)

# Cài đặt trục
ax.set_xlim(0, data.shape[1])
ax.set_ylim(0, data.shape[0])
ax.invert_yaxis()  # Đảo ngược trục Y để phù hợp với hệ tọa độ của ảnh

ax.axis('off')

# Lưu biểu đồ vào tệp hình ảnh
plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
plt.close()  # Đóng hình để giải phóng tài nguyên


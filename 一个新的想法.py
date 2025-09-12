import cv2
import numpy as np

file_path = "7505567186023497766\\000000.jpg"
img = cv2.imread(file_path)

if img is None:
    print("无法读取图像")
    exit()

original_height, original_width = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Sobel边缘检测而不是直接使用灰度图
sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=5)
sobel_y = cv2.convertScaleAbs(sobel_y)  # 转为8位
blurred = cv2.GaussianBlur(sobel_y, (5, 5), 0)
canny = cv2.Canny(blurred, 50, 150)
sobel_y = canny
cv2.imshow("sobel_y", sobel_y)

# 提取多列而不是仅第一列，提高鲁棒性
columns_to_check = 5  # 检查前5列
non_zero_indices_list = []

for col in range(columns_to_check):
    column_pixels = sobel_y[:, col]
    non_zero_indices = np.nonzero(column_pixels)[0]
    if len(non_zero_indices) > 0:
        non_zero_indices_list.append(non_zero_indices[0])  # 每列的第一个非零索引
        non_zero_indices_list.append(non_zero_indices[-1])  # 每列的最后一个非零索引

# 计算所有非零索引的最小值和最大值
if non_zero_indices_list:
    FstNoneZeroPixelIndex = min(non_zero_indices_list)
    LstNoneZeroPixelIndex = max(non_zero_indices_list)
else:
    # 如果没有找到非零像素，使用整个图像
    FstNoneZeroPixelIndex = 0
    LstNoneZeroPixelIndex = original_height - 1

print(f"上边界: {FstNoneZeroPixelIndex}")
print(f"下边界: {LstNoneZeroPixelIndex}")

# 截取区域
cropped_img = img[FstNoneZeroPixelIndex:LstNoneZeroPixelIndex, :]

# 显示结果
cv2.imshow("Original", img)
cv2.imshow("Cropped", cropped_img)

# 添加一些边界检查
if cropped_img.size == 0:
    print("警告: 截取区域为空")
else:
    print(f"截取区域尺寸: {cropped_img.shape}")

cv2.waitKey(0)
cv2.destroyAllWindows()
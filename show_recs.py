import cv2
import numpy as np

test_file_path = "7505567186023497766\\000000.jpg"
img = cv2.imread(test_file_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=5)
sobel_y = cv2.convertScaleAbs(sobel_y)  # 转为8位图像便于显示
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
erosion = cv2.erode(sobel_y, kernel, iterations=1)
cv2.imshow("erosion",erosion)
# 膨胀操作
dilation = cv2.dilate(erosion, kernel, iterations=1)
cv2.imshow("dilation",dilation)

# sobel_y = cv2.copyMakeBorder(sobel_y, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=255)
# sobel_y = cv2.copyMakeBorder(sobel_y, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=0)

# 缩放到原来的 1/3 大小
sobel_y_small = cv2.resize(dilation, (0, 0), fx=0.33, fy=0.33)
contours, _ = cv2.findContours(sobel_y_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_rect = None

# 获取原始图像大小
original_height, original_width = img.shape[:2]
original_area = original_width * original_height

# 计算缩放后的图像大小
scaled_height, scaled_width = sobel_y_small.shape[:2]
scaled_area = scaled_width * scaled_height

# 筛选矩形轮廓
for cnt in contours:
    # 计算轮廓周长
    perimeter = cv2.arcLength(cnt, True)
    # 轮廓近似，epsilon为近似精度，通常为周长的1%-5%
    approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
    
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(approx)
    original_rect_area = (w / 0.33) * (h / 0.33)
    temp_img = img.copy()
    # cv2.imshow("cnt_picture",cv2.rectangle(cv2.resize(temp_img,(0, 0), fx=0.33, fy=0.33), (x, y), (x+w, y+h), (0, 255, 0), 2))
    # cv2.waitKey(0)
    print(original_area)
    if original_rect_area < original_area and original_rect_area > original_area * 0.1:
        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)

    # 检查是否为四边形（矩形有4个顶点）
    # if len(approx) == 4:
    #     # 计算轮廓面积
    #     area = cv2.contourArea(cnt)
    #     # 检查面积是否足够大，排除小的噪点轮廓
    #     if area > 100:
    #         # 计算边界矩形
    #         x, y, w, h = cv2.boundingRect(approx)
            
    #         # 检查宽高比，避免极端比例的形状
    #         aspect_ratio = float(w) / h
    #         if 0.5 <= aspect_ratio <= 2.0:
    #             # 计算缩放前的面积
    #             original_rect_area = (w / 0.33) * (h / 0.33)
                
    #             # 检查矩形面积是否小于原图像大小且大于原图像大小的50%
    #             if original_rect_area < original_area and original_rect_area > original_area * 0.1:
    #                 # 更新最大面积和最佳矩形
    #                 if area > max_area:
    #                     max_area = area
    #                     best_rect = (x, y, w, h)

# 显示最大矩形轮廓
if best_rect is not None:
    x, y, w, h = best_rect
    img_with_cnt = cv2.resize(img, (0, 0), fx=0.33, fy=0.33).copy()
    cv2.rectangle(img_with_cnt, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Largest Rectangle Contour', img_with_cnt)
    cv2.waitKey(0)
    
    # 还原到原始图像大小
    x_original = int(x / 0.33)
    y_original = int(y / 0.33)
    w_original = int(w / 0.33)
    h_original = int(h / 0.33)
    
    # 截取区域
    extracted_region = img[y_original:y_original+h_original, x_original:x_original+w_original]
    
    # 显示截取的区域
    cv2.imshow('Extracted Rectangle', extracted_region)
    cv2.waitKey(0)
else:
    print("No suitable rectangle found that meets the size criteria")

cv2.destroyAllWindows()
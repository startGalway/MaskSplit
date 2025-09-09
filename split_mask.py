import cv2
import numpy as np

img = cv2.imread(r'7505567186023497766\000000.jpg')

# 1. 灰度化并高斯模糊，减少噪声
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. 边缘检测
edges = cv2.Canny(blur, 50, 150)

# 3. 闭操作填补边缘空洞
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 4. 查找轮廓，筛选最大矩形区域（假设主体区域最大）
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_area = 0
best_rect = None
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > max_area:
        max_area = area
        best_rect = (x, y, w, h)

# 5. 截取最大矩形区域（主体）
if best_rect is not None:
    x, y, w, h = best_rect
    subject = img[y:y+h, x:x+w]
    cv2.imwrite('subject_rect.jpg', subject)
    cv2.imshow('主体区域', subject)
else:
    print("未找到主体区域")

cv2.waitKey(0)
cv2.destroyAllWindows()
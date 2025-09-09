import cv2
test_file_path = "7505567186023497766\\000000.jpg"
img = cv2.imread(test_file_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=5)
sobel_y = cv2.convertScaleAbs(sobel_y)  # 转为8位图像便于显示
sobel_y = cv2.copyMakeBorder(sobel_y, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=255)
sobel_y = cv2.copyMakeBorder(sobel_y, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=0)


cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)
# 缩放到原来的 1/3 大小
sobel_y_small = cv2.resize(sobel_y, (0, 0), fx=0.33, fy=0.33)
contours, _ = cv2.findContours(sobel_y_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('Sobel Y', sobel_y_small)

max_area = 0
best_rect = None
for cnt in contours:
    img_with_cnt = cv2.resize(img, (0, 0), fx=0.33, fy=0.33).copy()
    cv2.drawContours(image=img_with_cnt, contours=[cnt], contourIdx=-1, color=128, thickness=2)
    cv2.imshow('contour', img_with_cnt)
    cv2.waitKey(0)
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > max_area:
        max_area = area
        best_rect = (x, y, w, h)

# 5. 截取最大矩形区域（主体）




cv2.waitKey(0)
cv2.destroyAllWindows()
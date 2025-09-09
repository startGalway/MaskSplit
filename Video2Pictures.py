import cv2
import os

video_path = "7505567186023497766.mp4"

# 创建同名文件夹（去掉扩展名）
output_dir = os.path.splitext(video_path)[0]
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_name = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
    cv2.imwrite(frame_name, frame)
    frame_idx += 1

cap.release()
print(f"共导出 {frame_idx} 帧到文件夹 {output_dir}")


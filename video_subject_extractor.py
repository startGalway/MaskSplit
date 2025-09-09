import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip
class SubjectExtractor:
    def __init__(self, video_path, output_video_path, temp_dir='temp_frames'):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_subject_rect(self, img):
        # 先加一圈蒙版色边框（如橙色，BGR: 0,128,255）
        border_color = (0, 128, 255)  # 你可以根据实际蒙版色调整
        img_with_border = cv2.copyMakeBorder(
            img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=border_color
        )

        gray = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_rect = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)
        if best_rect is not None:
            x, y, w, h = best_rect
            # 去掉边框的偏移
            x = max(x - 20, 0)
            y = max(y - 20, 0)
            return img[y:y+h-40, x:x+w-40]  # 注意这里要减去上下左右各20像素
        else:
            return None

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_list = []
        idx = 0
        min_w, min_h = None, None

        # 先处理所有帧，找出主体区域的最小宽高，保证拼接时尺寸一致
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            subject = self.extract_subject_rect(frame)
            if subject is not None:
                h, w = subject.shape[:2]
                if min_w is None or w < min_w:
                    min_w = w
                if min_h is None or h < min_h:
                    min_h = h
                frame_list.append(subject)
            idx += 1
        cap.release()

        # 统一裁剪尺寸
        resized_frames = []
        for i, f in enumerate(frame_list):
            h, w = f.shape[:2]
            crop = f[0:min_h, 0:min_w]
            frame_path = os.path.join(self.temp_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, crop)
            resized_frames.append(crop)

        # 写入新视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (min_w, min_h))
        for f in resized_frames:
            out.write(f)
        out.release()

    def add_audio(self):
        # 用moviepy合成音视频
        video_clip = VideoFileClip('temp_video.mp4')
        audio_clip = AudioFileClip(self.video_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(self.output_video_path, codec='libx264', audio_codec='aac')
        video_clip.close()
        audio_clip.close()

    def run(self):
        self.process_video()
        self.add_audio()
        print(f"处理完成，输出视频：{self.output_video_path}")

if __name__ == "__main__":
    extractor = SubjectExtractor(
        video_path="7505567186023497766.mp4",
        output_video_path="output_subject_video.mp4"
    )
    extractor.run()
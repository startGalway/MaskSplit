import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip


class VideoFrameCropper:
    def __init__(self):
        self.original_height = 0
        self.original_width = 0
        self.crop_top = 0
        self.crop_bottom = 0

    def process_frame(self, frame):
        """处理单帧图像，返回裁剪后的图像"""
        if frame is None:
            return None

        self.original_height, self.original_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用Sobel边缘检测
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=5)
        sobel_y = cv2.convertScaleAbs(sobel_y)  # 转为8位
        blurred = cv2.GaussianBlur(sobel_y, (5, 5), 0)
        canny = cv2.Canny(blurred, 50, 150)
        sobel_y = canny

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
            LstNoneZeroPixelIndex = self.original_height - 1

        self.crop_top = FstNoneZeroPixelIndex + 10
        self.crop_bottom = LstNoneZeroPixelIndex - 10

    def process_video(self, input_video_path, output_video_path):
        """处理整个视频文件"""
        # 检查输入文件是否存在
        if not os.path.exists(input_video_path):
            print(f"错误: 输入视频文件不存在 {input_video_path}")
            return False

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 使用OpenCV打开视频
        cap = cv2.VideoCapture(input_video_path)

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")

        # 读取第一帧以确定裁剪尺寸
        ret, first_frame = cap.read()
        if not ret:
            print("错误: 无法读取视频帧")
            cap.release()
            return False

        self.process_frame(first_frame)

        # 重置视频到第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 定义视频编码器和输出
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width
        out_height = self.crop_bottom - self.crop_top
        temp_output_path = "temp_video_no_audio.mp4"
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_width, out_height))

        # 处理每一帧
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 裁剪帧
            cropped_frame = frame[self.crop_top:self.crop_bottom, :]
            out.write(cropped_frame)

            # 显示进度
            if frame_count % 30 == 0:
                print(f"处理进度: {frame_count}/{total_frames} 帧")

            frame_count += 1

        # 释放资源
        cap.release()
        out.release()

        # 使用MoviePy提取音频并合并到处理后的视频
        try:
            # 提取原视频音频
            video_clip = VideoFileClip(input_video_path)
            audio_clip = video_clip.audio

            # 加载处理后的视频（无音频）
            video_clip_processed = VideoFileClip(temp_output_path)

            # 添加音频到处理后的视频
            final_clip = video_clip_processed.set_audio(audio_clip)

            # 输出最终视频
            final_clip.write_videofile(
                output_video_path,
                codec='libx264',
                audio_codec='aac',
                fps=fps
            )

            # 关闭所有剪辑
            video_clip.close()
            video_clip_processed.close()
            final_clip.close()

            # 删除临时文件
            os.remove(temp_output_path)

        except Exception as e:
            print(f"处理音频时出错: {e}")
            return False

        print(f"视频处理完成: {output_video_path}")
        return True


def main():
    # 使用示例
    cropper = VideoFrameCropper()

    # 指定输入和输出视频路径
    input_video = "7409295710242684938.mp4"  # 替换为您的视频路径
    output_video = "output_video.mp4"  # 替换为您想要的输出路径

    # 处理视频
    success = cropper.process_video(input_video, output_video)

    if success:
        print("视频处理成功完成!")
    else:
        print("视频处理失败!")


if __name__ == "__main__":
    main()
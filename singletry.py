import cv2
import numpy as np

def extract_video_center(video_path, output_path, margin=0, min_content_height_ratio=0.1):
    """
    从上下背景板的视频中提取中间主体部分

    Args:
        video_path (str): 输入视频路径
        output_path (str): 输出视频路径
        margin (int, optional): 在裁剪区域上下额外保留的像素，避免裁剪太紧。默认为0。
        min_content_height_ratio (float, optional): 判断为内容区域的最小高度与视频高度的比值，用于过滤噪声。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 选取中间一帧作为参考帧来分析内容区域 (避免片头片尾可能无内容)
    sample_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
    ret, sample_frame = cap.read()
    if not ret:
        print("Error: Could not read sample frame.")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置回第一帧

    # 转换为灰度图
    gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)

    # 计算垂直方向的Sobel边缘梯度（绝对值）
    # 这有助于突出垂直方向的变化（通常是内容与背景的分界）
    sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_vertical_abs = np.abs(sobel_vertical)
    sobel_vertical_abs = np.uint8(sobel_vertical_abs / np.max(sobel_vertical_abs) * 255)

    # 垂直投影：计算每行像素的平均梯度值
    vertical_projection = np.mean(sobel_vertical_abs, axis=1)

    # 平滑投影曲线，减少噪声影响
    kernel_size = max(5, int(orig_height * 0.05) | 1)  # 确保是奇数
    vertical_projection_smooth = cv2.GaussianBlur(vertical_projection, (kernel_size, 1), 0)

    # 动态阈值：寻找梯度值较高的区域（内容区域）
    # 阈值设为平滑后投影的最大值的一部分
    threshold_val = np.max(vertical_projection_smooth) * 0.2
    content_mask = vertical_projection_smooth > threshold_val

    # 找出内容区域的起始行和结束行
    content_indices = np.where(content_mask)[0]
    if content_indices.size > 0:
        top_content = max(0, content_indices[0] - margin)
        bottom_content = min(orig_height - 1, content_indices[-1] + margin)
        crop_height = bottom_content - top_content + 1

        # 安全检查：确保裁剪高度是合理的（例如，大于视频高度的10%）
        min_crop_height = int(orig_height * min_content_height_ratio)
        if crop_height < min_crop_height:
            print(f"Warning: Calculated content height ({crop_height}px) is too small. Using fallback: center cropping with 80% height.")
            top_content = int(orig_height * 0.1)
            bottom_content = int(orig_height * 0.9)
            crop_height = bottom_content - top_content
    else:
        # 如果没有找到明显的内容区域，则使用回退方案：裁剪中间大部分区域
        print("Warning: Could not clearly detect content region. Using fallback: center cropping with 80% height.")
        top_content = int(orig_height * 0.1)
        bottom_content = int(orig_height * 0.9)
        crop_height = bottom_content - top_content

    print(f"Original height: {orig_height}px")
    print(f"Calculated content region: from row {top_content} to {bottom_content} (height: {crop_height}px)")

    # 定义视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, crop_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 裁剪当前帧
        cropped_frame = frame[top_content:bottom_content + 1, :]
        out.write(cropped_frame)

        frame_count += 1
        # 可选：打印进度
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Finished! Cropped video saved to: {output_path}")

# 使用示例
if __name__ == "__main__":
    input_video = r"C:\Users\Think\Desktop\MaskSplit\OutPutMovies\7349200461708361791.mp4"  # 替换为你的输入视频路径
    output_video = "extracted_center_video.mp4" # 输出视频路径
    extract_video_center(input_video, output_video, margin=5) # margin可调整，避免裁剪太紧
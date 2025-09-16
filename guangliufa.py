import cv2
import numpy as np


def find_static_background_region(video_path, output_path=None, roi_height_ratio=0.2):
    """
    使用光流法检测视频中最不活跃的横向区域（背景板）

    Args:
        video_path (str): 输入视频文件路径
        output_path (str, optional): 输出视频文件路径（用于可视化调试）. Defaults to None.
        roi_height_ratio (float, optional): 假设的背景板区域大致占屏幕高度的比例，用于辅助定位。默认0.2（即20%）。
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # 获取视频第一帧来初始化
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return None

    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 初始化一个数组来累积每行的运动活跃度（可选：也可以使用移动平均）
    row_activity_accumulator = np.zeros(height, dtype=np.float32)
    frame_count = 0

    #  Farneback 光流法的参数 [2](@ref)
    pyr_scale = 0.5  # 图像金字塔缩放比例
    levels = 3  # 金字塔层数
    winsize = 15  # 平均窗口大小
    iterations = 3  # 迭代次数
    poly_n = 5  # 像素邻域大小
    poly_sigma = 1.2  # 高斯标准差
    flags = 0  # 可选标志

    # 如果提供输出路径，则初始化VideoWriter用于可视化调试
    if output_path:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        # 修改为合并后帧的尺寸（宽度*2，高度）
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    else:
        out = None

    # 确定我们关心的背景区域高度（例如，顶部和底部各20%）
    roi_top_height = int(height * roi_height_ratio)
    roi_bottom_height = height - int(height * roi_height_ratio)
    candidate_heights = [roi_top_height, roi_bottom_height]  # 通常背景板在顶部或底部

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流 [2](@ref)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, np.zeros_like(prev_gray),
                                            pyr_scale, levels, winsize, iterations,
                                            poly_n, poly_sigma, flags)

        # 计算光流幅度 [2](@ref)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 计算每一行的平均运动幅度
        row_activity = np.mean(magnitude, axis=1)  # 这是一个一维数组，长度为height

        # 累积运动幅度（简单累加，后续可取平均）
        row_activity_accumulator += row_activity
        frame_count += 1

        # --- 可视化部分（可选，用于调试和理解过程）---
        if out is not None:
            # 将运动幅度归一化用于显示
            mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            mag_normalized = mag_normalized.astype(np.uint8)
            mag_colored = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_HOT)  # 热力图更直观

            # 在当前帧上绘制最不活跃的行（假设是背景）
            min_active_row = np.argmin(row_activity)  # 找到当前帧最不活跃的行
            cv2.line(frame, (0, min_active_row), (width - 1, min_active_row), (0, 255, 0), 2)

            # 将光流可视化图和原图合并显示
            vis_frame = np.hstack((frame, mag_colored))
            cv2.putText(vis_frame, "Green line: Least active row (potential background)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(vis_frame)
        # --- 可视化结束 ---

        prev_gray = frame_gray.copy()

    cap.release()
    if out:
        out.release()

    # 计算整个视频序列中每行的平均运动活跃度
    avg_row_activity = row_activity_accumulator / frame_count

    # 寻找最不活跃的连续区域（这里以顶部背景板为例，寻找顶部候选区域内最不活跃的行）
    top_region_activity = avg_row_activity[:roi_top_height]
    bottom_region_activity = avg_row_activity[roi_bottom_height:]

    # 假设背景板要么在顶部，要么在底部，比较两个区域的平均活跃度
    avg_activity_top = np.mean(top_region_activity) if len(top_region_activity) > 0 else float('inf')
    avg_activity_bottom = np.mean(bottom_region_activity) if len(bottom_region_activity) > 0 else float('inf')

    if avg_activity_top <= avg_activity_bottom:
        background_type = "top"
        # 在顶部区域中找到平均运动幅度最小的行
        background_row_start = 0
        background_row_end = np.argmin(top_region_activity)  # 这只是最小值的索引，可能需要一个范围
        print(f"Background is likely at the TOP. Least active row index: {background_row_end}")
        # 更稳健的方法是返回一个区域 [start_y, end_y]
        background_region = (0, background_row_end)
    else:
        background_type = "bottom"
        # 在底部区域中找到平均运动幅度最小的行（注意索引偏移）
        min_index_in_bottom_region = np.argmin(bottom_region_activity)
        background_row_start = roi_bottom_height + min_index_in_bottom_region
        background_row_end = height - 1
        print(f"Background is likely at the BOTTOM. Least active row index: {background_row_start}")
        background_region = (background_row_start, height - 1)

    print(f"Average activity in top region: {avg_activity_top}")
    print(f"Average activity in bottom region: {avg_activity_bottom}")

    return background_region, background_type, avg_row_activity


# 使用示例
if __name__ == "__main__":
    input_video = r"C:\Users\Think\Desktop\MaskSplit\OutPutMovies\7349200461708361791.mp4"
    output_debug_video = "debug_optical_flow_analysis.mp4"  # 可选：用于查看处理过程

    bg_region, bg_type, activity_data = find_static_background_region(input_video, output_debug_video)

    if bg_region:
        print(f"\nEstimated static background region (Y-range): {bg_region}")
        # 现在你可以根据bg_region来裁剪你的视频，移除背景板
        # 例如，如果背景板在顶部 (0, 100)，那么你想要的主体区域就是从 y=100 到 height-1optical_flow(r"C:\Users\Think\Desktop\MaskSplit\OutPutMovies\7349200461708361791.mp4", "output_flow.mp4")
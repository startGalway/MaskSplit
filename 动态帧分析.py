import os
import sys
import json
import argparse
from typing import List

import cv2
import numpy as np


def sample_first_frames_each_second(video_path: str, max_seconds: int | None = None) -> tuple[list[np.ndarray], list[np.ndarray], float, tuple[int, int]]:
    """
    从视频中每秒抽取首帧。
    返回：
    - frames_bgr: List[BGR帧]
    - frames_gray: List[gray帧]
    - fps: 帧率
    - (width, height): 尺寸
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if frame_count == 0 or width == 0 or height == 0:
        cap.release()
        raise RuntimeError("视频元数据无效，无法读取帧。")

    total_seconds = int(np.floor(frame_count / max(fps, 1e-6)))
    if max_seconds is not None:
        total_seconds = min(total_seconds, max_seconds)
    # 至少采样 1 帧，处理 <1s 短视频
    total_seconds = max(1, total_seconds)

    frames_bgr: list[np.ndarray] = []
    frames_gray: list[np.ndarray] = []

    for s in range(total_seconds):
        frame_idx = int(round(s * fps))
        # 钳制到范围内
        frame_idx = max(0, min(frame_idx, frame_count - 1))
        # 定位到该秒的首帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames_bgr.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)

    cap.release()

    if len(frames_gray) == 0:
        raise RuntimeError("未能成功抽取任何帧。")

    return frames_bgr, frames_gray, float(fps), (width, height)


def compute_background_median(frames_gray: List[np.ndarray]) -> np.ndarray:
    """基于采样灰度序列计算逐像素时域中位数，得到背景图。"""
    stack = np.stack(frames_gray, axis=0).astype(np.uint8)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg


def compute_row_scores(frames_gray: List[np.ndarray], bg_gray: np.ndarray) -> np.ndarray:
    """
    计算每一行的稳定度分数（越小越稳定）。
    方法：对每帧与背景的绝对差，按行求均值，再对时间求 90 分位。
    输出范围约为 [0, 1]（通过 255 归一化）。
    """
    H, W = bg_gray.shape[:2]  # 获取背景图的高和宽
    if any(f.shape[:2] != (H, W) for f in frames_gray):
        raise ValueError("帧尺寸与背景不一致。")  # 检查所有帧尺寸是否一致

    diffs: list[np.ndarray] = []  # 存储每帧与背景的行均值差异
    bg_f32 = bg_gray.astype(np.float32)  # 背景图转为 float32 类型
    for f in frames_gray:
        d = np.abs(f.astype(np.float32) - bg_f32) / 255.0  # 计算每帧与背景的归一化绝对差
        row_mean = d.mean(axis=1)  # 按行求均值，得到每行的平均差异
        diffs.append(row_mean)  # 加入列表

    diffs_stack = np.stack(diffs, axis=0)  # 堆叠为二维数组，形状为 (帧数, 行数)
    row_scores = np.percentile(diffs_stack, 90, axis=0)  # 对每行取 90 分位，反映稳定度
    return row_scores.astype(np.float32)  # 返回每行分数，float32 类型


def choose_threshold(row_scores: np.ndarray) -> float:
    """自适应阈值：以中位数为基，限制在 [0.02, 0.1] 区间。"""
    med = float(np.median(row_scores))
    thr = med * 1.5
    thr = max(0.02, min(0.10, thr))
    return thr


# 对一维布尔向量做形态学闭运算，填补小孔洞（如微小抖动/噪声造成的不稳定行）
def _close_bool_1d(stable: np.ndarray, k: int) -> np.ndarray:
    """对稳定掩码做 1D 闭运算；k 为核高（行数）。"""
    if k <= 1:
        return stable.copy()
    col = (stable.astype(np.uint8) * 255).reshape(-1, 1)
    kernel = np.ones((int(k), 1), np.uint8)
    closed = cv2.morphologyEx(col, cv2.MORPH_CLOSE, kernel)
    return (closed.reshape(-1) > 0)


# 从底部向上，允许跨过一次“字幕间隙”寻找更靠近主体的下背景上边界
def _find_bottom_edge_allow_one_gap(stable: np.ndarray, min_bottom_len: int, max_skip_gap: int) -> int:
    """
    返回下背景的最上边界索引（含）。
    策略：
    1) 先找到最底下的一段稳定区（长度>=min_bottom_len），得到 start1。
    2) 若其上方存在一段不稳定区，且长度<=max_skip_gap，则再往上寻找第二段稳定区（长度>=min_bottom_len），若存在则返回该段的起点 start2；
       否则返回 start1。
    3) 若第一段都不存在，返回 H（表示无）。
    """
    H = int(stable.shape[0])
    i = H - 1

    # 第一段稳定区（靠近底部）
    if i < 0:
        return H
    end1 = i
    while i >= 0 and stable[i]:
        i -= 1
    start1 = i + 1
    L1 = end1 - start1 + 1 if end1 >= start1 else 0
    if L1 < min_bottom_len:
        return H

    # 间隙（可能是字幕）
    end_gap = i
    while i >= 0 and (not stable[i]):
        i -= 1
    start_gap = i + 1
    G = end_gap - start_gap + 1 if end_gap >= start_gap else 0

    if G > 0 and G <= max_skip_gap:
        # 第二段稳定区（字幕之上、靠近主体）
        end2 = i
        while i >= 0 and stable[i]:
            i -= 1
        start2 = i + 1
        L2 = end2 - start2 + 1 if end2 >= start2 else 0
        if L2 >= min_bottom_len:
            return start2

    return start1


def extend_run_from_top(stable: np.ndarray, max_gap: int = 3) -> int:
    """从顶部（0）向下扩展稳定区域，允许最多 max_gap 的非稳定行间隙。返回最底部索引，若不存在则返回 -1。"""
    gap = 0
    top_end = -1
    for i, v in enumerate(stable):
        if v:
            top_end = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return top_end


def extend_run_from_bottom(stable: np.ndarray, max_gap: int = 3, skip_first_unstable: bool = False) -> int:
    """从底部（H-1）向上扩展稳定区域。
    - skip_first_unstable=False：保持原逻辑，允许最多 max_gap 的非稳定行间隙，返回该段稳定区的最上方行索引；若不存在返回 H。
    - skip_first_unstable=True：越过底部第一段不稳定区（字幕），在其上的下一段稳定区内向上扩展，返回这段稳定区的最上方行索引；若不存在返回 H。
    """
    H = int(stable.shape[0])
    if H == 0:
        return 0

    if not skip_first_unstable:
        gap = 0
        bottom_start = H
        for i in range(H - 1, -1, -1):
            if stable[i]:
                bottom_start = i
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        return bottom_start

    # 跨越第一次不稳定区（字幕）的逻辑
    i = H - 1

    # phase A：越过底部的稳定区（若有）
    while i >= 0 and stable[i]:
        i -= 1

    # phase B：越过第一段不稳定区（字幕）
    saw_unstable = False
    while i >= 0 and (not stable[i]):
        saw_unstable = True
        i -= 1

    if not saw_unstable or i < 0:
        return H

    # phase C：第二段稳定区（字幕之上、靠近主体），向上扩展并记录起点
    bottom_start = i
    while i >= 0 and stable[i]:
        bottom_start = i
        i -= 1
    return bottom_start


def detect_background_bands(row_scores: np.ndarray, threshold: float, min_top_len: int = 5, min_bottom_len: int = 5) -> tuple[int, int, np.ndarray]:
    """
    根据行分数与阈值检测主体区域：直接提取“最长的一段连续不稳定区域”。
    返回：UpperAreaLowestEdge, LowerAreaUppestEdge, stable_mask
    - UpperAreaLowestEdge: 主体上边界的上一行（若主体从第 0 行开始，则为 -1）
    - LowerAreaUppestEdge: 主体下边界的下一行（若主体到达 H-1 行，则为 H）
    - stable_mask: 布尔数组，表示稳定行（True 为稳定）
    """
    H = int(row_scores.shape[0])
    stable = (row_scores <= threshold)
    unstable = ~stable

    # 若不存在任何不稳定行，则不裁剪（返回整幅）
    if not np.any(unstable):
        return -1, H, stable

    # 寻找最长连续的不稳定段，若长度相同，选择更靠近中线者
    best_start = 0
    best_end = -1
    best_len = 0
    center = (H - 1) / 2.0

    i = 0
    while i < H:
        if not unstable[i]:
            i += 1
            continue
        j = i
        while j + 1 < H and unstable[j + 1]:
            j += 1
        length = j - i + 1
        if length > best_len:
            best_len = length
            best_start, best_end = i, j
        elif length == best_len and best_len > 0:
            cur_center_dist = abs((i + j) / 2.0 - center)
            best_center_dist = abs((best_start + best_end) / 2.0 - center)
            if cur_center_dist < best_center_dist:
                best_start, best_end = i, j
        i = j + 1

    # 将主体段 [best_start, best_end] 转换为裁剪接口所需的边界：
    # 上边界返回为主体上边的上一行；下边界返回为主体下边的下一行
    upper_edge = best_start - 1 if best_start > 0 else -1
    lower_edge = best_end + 1 if (best_end + 1) <= H else H

    return upper_edge, lower_edge, stable


def crop_video_vertical(in_path: str, out_path: str, y_start: int, y_end: int) -> tuple[int, int]:
    """
    将视频沿纵向裁剪到 [y_start, y_end) 区间。
    返回：(写出的帧数，总帧数)
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    y_start = max(0, min(y_start, height))
    y_end = max(0, min(y_end, height))
    if y_end <= y_start:
        # 不裁剪，直接复制
        y_start, y_end = 0, height

    crop_h = y_end - y_start
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 使用 mp4v 编码
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开输出视频写入器: {out_path}")

    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y_start:y_end, :, :]
        writer.write(crop)
        written += 1

    cap.release()
    writer.release()
    return written, total


def save_debug_visual(frame_bgr: np.ndarray, upper_edge: int, lower_edge: int, out_path: str) -> None:
    """在参考帧上绘制上下边界并保存。"""
    vis = frame_bgr.copy()
    H, W = vis.shape[:2]
    # 绘制线
    if upper_edge >= 0:
        cv2.line(vis, (0, upper_edge), (W - 1, upper_edge), (0, 255, 0), 2)
    if lower_edge < H:
        cv2.line(vis, (0, lower_edge), (W - 1, lower_edge), (0, 0, 255), 2)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, vis)


def analyze_and_crop(
        input_video: str,
        output_video: str | None = None,
        output_json: str | None = None,
        debug_image: str | None = None,
        max_seconds: int | None = None,
        analyze_only: bool = False,
    ) -> dict:
        """
        主流程：采样 -> 背景中位数 -> 行分数 -> 阈值 -> 边界 -> 可选裁剪。
        返回结果字典，包含上下边界、阈值、统计信息等。
        """
        # 采样视频，每秒抽取首帧，得到彩色帧和灰度帧，以及帧率和尺寸
        frames_bgr, frames_gray, fps, (W, H) = sample_first_frames_each_second(input_video, max_seconds=max_seconds)

        # 计算所有灰度帧的逐像素中位数，得到背景图
        bg = compute_background_median(frames_gray)
        # 计算每一行的稳定度分数
        row_scores = compute_row_scores(frames_gray, bg)
        # 根据分数自适应选择阈值
        threshold = choose_threshold(row_scores)

        # 检测上下背景区域边界和稳定行掩码
        upper_edge, lower_edge, stable_mask = detect_background_bands(row_scores, threshold)

        # 构造结果字典，包含输入、统计、边界等信息
        result = {
            "input": input_video,
            "fps": fps,
            "width": W,
            "height": H,
            "num_sampled_frames": len(frames_gray),
            "threshold": float(threshold),
            "UpperAreaLowestEdge": int(upper_edge),
            "LowerAreaUppestEdge": int(lower_edge),
            "stable_rows_top": list(range(0, upper_edge + 1)) if upper_edge >= 0 else [],
            "stable_rows_bottom": list(range(lower_edge, H)) if lower_edge < H else [],
        }

        # 如果指定了调试图片，保存带边界线的可视化图
        if debug_image:
            try:
                save_debug_visual(frames_bgr[0], upper_edge, lower_edge, debug_image)
                result["debug_image"] = debug_image
            except Exception as e:
                result["debug_image_error"] = str(e)

        # 如果指定了输出 JSON，保存分析结果
        if output_json:
            out_dir = os.path.dirname(output_json)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)  # type: ignore[arg-type]

        # 如果不是仅分析，则进行视频裁剪
        if not analyze_only:
            if output_video is None:
                base, ext = os.path.splitext(os.path.basename(input_video))
                output_video = os.path.join(os.path.dirname(input_video), f"{base}_cropped{ext or '.mp4'}")
            # 计算裁剪范围：开区间 (upper_edge, lower_edge) -> 半开区间 [upper_edge+1, lower_edge)
            y0 = (upper_edge + 1) if upper_edge >= 0 else 0
            y1 = lower_edge if lower_edge <= H else H
            try:
                # 执行裁剪，返回写入帧数和总帧数
                written, total = crop_video_vertical(input_video, output_video, y0, y1)
                result["output_video"] = output_video
                result["cropped_rows"] = [int(y0), int(y1)]
                result["frames_written"] = written
                result["frames_total"] = total
            except Exception as e:
                result["crop_error"] = str(e)

        # 返回结果字典
        return result


def build_default_paths(input_video: str) -> tuple[str, str]:
    base = os.path.splitext(os.path.basename(input_video))[0]
    out_dir = os.path.join(os.path.dirname(input_video), "动态帧分析输出")
    json_path = os.path.join(out_dir, f"{base}_analysis.json")
    debug_img = os.path.join(out_dir, f"{base}_debug.png")
    return json_path, debug_img


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按秒采样并检测上下背景行，自动纵向裁剪视频")
    p.add_argument("--input", required=True, help="输入视频路径")
    p.add_argument("--output", default=None, help="输出裁剪视频路径（可选）")
    p.add_argument("--json", default=None, help="输出分析 JSON 路径（可选）")
    p.add_argument("--debug-image", default=None, help="输出调试图片路径（可选）")
    p.add_argument("--max-seconds", type=int, default=None, help="采样的最长秒数，限制计算量（可选）")
    p.add_argument("--analyze-only", action="store_true", help="仅分析不裁剪")
    return p.parse_args(argv)


def main() -> int:
    # 直接运行配置（无需命令行）
    input_video = r"C:\Users\Think\Desktop\校验视频\1206-1406\1326-1365\7435955091922337802.mp4"
    output_video = "动态分析裁剪.mp4"
    max_seconds = None
    analyze_only = False
    json_path = None
    debug_img = None

    if not os.path.exists(input_video):
        print(f"输入视频不存在: {input_video}", file=sys.stderr)
        return 2

    if json_path is None or debug_img is None:
        # 提供默认输出位置
        default_json, default_debug = build_default_paths(input_video)
        if json_path is None:
            json_path = default_json
        if debug_img is None:
            debug_img = default_debug

    try:
        result = analyze_and_crop(
            input_video=input_video,
            output_video=output_video,
            output_json=json_path,
            debug_image=debug_img,
            max_seconds=max_seconds,
            analyze_only=analyze_only,
        )
    except Exception as e:
        print(f"处理失败: {e}", file=sys.stderr)
        return 1

    # 控制台输出关键信息
    print(json.dumps({
        "UpperAreaLowestEdge": result.get("UpperAreaLowestEdge"),
        "LowerAreaUppestEdge": result.get("LowerAreaUppestEdge"),
        "threshold": result.get("threshold"),
        "stable_rows_top_count": len(result.get("stable_rows_top", [])),
        "stable_rows_bottom_count": len(result.get("stable_rows_bottom", [])),
        "output_video": result.get("output_video"),
        "cropped_rows": result.get("cropped_rows"),
        "debug_image": result.get("debug_image"),
        "json": json_path,
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
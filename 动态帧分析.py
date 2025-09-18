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
    H, W = bg_gray.shape[:2]
    if any(f.shape[:2] != (H, W) for f in frames_gray):
        raise ValueError("帧尺寸与背景不一致。")

    diffs: list[np.ndarray] = []
    bg_f32 = bg_gray.astype(np.float32)
    for f in frames_gray:
        d = np.abs(f.astype(np.float32) - bg_f32) / 255.0  # HxW   对每帧和背景图的每一个像素的差异做归一化
        row_mean = d.mean(axis=1)  # H   按行求均值
        diffs.append(row_mean)

    diffs_stack = np.stack(diffs, axis=0)  # TxH
    row_scores = np.percentile(diffs_stack, 90, axis=0)  # H
    return row_scores.astype(np.float32)


def choose_threshold(row_scores: np.ndarray) -> float:
    """自适应阈值：以中位数为基，限制在 [0.02, 0.1] 区间。"""
    med = float(np.median(row_scores))
    thr = med * 1.5
    thr = max(0.02, min(0.10, thr))
    return thr


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


def extend_run_from_bottom(stable: np.ndarray, max_gap: int = 3) -> int:
    """从底部（H-1）向上扩展稳定区域，允许最多 max_gap 的非稳定行间隙。返回区域起始索引（最上方稳定行），若不存在则返回数组长度。"""
    H = stable.shape[0]
    gap = 0
    bottom_start = H
    for i in range(H - 1, -1, -1):
        v = bool(stable[i])
        if v:
            bottom_start = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return bottom_start


def detect_background_bands(row_scores: np.ndarray, threshold: float, min_top_len: int = 5, min_bottom_len: int = 5) -> tuple[int, int, np.ndarray]:
    """
    根据行分数与阈值检测上下背景区域。
    返回：UpperAreaLowestEdge, LowerAreaUppestEdge, stable_mask
    - UpperAreaLowestEdge: 顶部背景区域的最下边界（含），若无则为 -1
    - LowerAreaUppestEdge: 底部背景区域的最上边界（含），若无则为 H
    - stable_mask: 布尔数组，表示稳定行
    """
    H = row_scores.shape[0]
    stable = row_scores <= threshold

    # 顶部
    top_end = extend_run_from_top(stable, max_gap=3)
    if top_end >= 0:
        # 保证长度
        if top_end + 1 < min_top_len:
            top_end = -1

    # 底部
    bottom_start = extend_run_from_bottom(stable, max_gap=3)
    if bottom_start < H:
        if (H - bottom_start) < min_bottom_len:
            bottom_start = H

    return top_end, bottom_start, stable


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
    frames_bgr, frames_gray, fps, (W, H) = sample_first_frames_each_second(input_video, max_seconds=max_seconds)

    bg = compute_background_median(frames_gray)
    row_scores = compute_row_scores(frames_gray, bg)
    threshold = choose_threshold(row_scores)

    upper_edge, lower_edge, stable_mask = detect_background_bands(row_scores, threshold)

    # JSON 结果
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

    # 调试图
    if debug_image:
        try:
            save_debug_visual(frames_bgr[0], upper_edge, lower_edge, debug_image)
            result["debug_image"] = debug_image
        except Exception as e:
            result["debug_image_error"] = str(e)

    # 保存 JSON
    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # 裁剪
    if not analyze_only:
        if output_video is None:
            base, ext = os.path.splitext(os.path.basename(input_video))
            output_video = os.path.join(os.path.dirname(input_video), f"{base}_cropped{ext or '.mp4'}")
        # 裁剪范围：开区间 (upper_edge, lower_edge) -> 半开区间 [upper_edge+1, lower_edge)
        y0 = (upper_edge + 1) if upper_edge >= 0 else 0
        y1 = lower_edge if lower_edge <= H else H
        try:
            written, total = crop_video_vertical(input_video, output_video, y0, y1)
            result["output_video"] = output_video
            result["cropped_rows"] = [int(y0), int(y1)]
            result["frames_written"] = written
            result["frames_total"] = total
        except Exception as e:
            result["crop_error"] = str(e)

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
    input_video = r"C:\Users\Think\Desktop\校验视频\1-200\1-40\7353976599824302092.mp4"
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
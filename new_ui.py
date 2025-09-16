import os
import sys
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QUrl, QRectF, QPointF, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QStyle,
    QSlider,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QSizePolicy,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

import cv2

# 导入现有处理逻辑
import new as processing_mod


class CropSelectorWidget(QWidget):
    """显示首帧图，并提供上下拖动的裁剪线（仅支持垂直方向的裁剪）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._pixmap: Optional[QPixmap] = None
        self._img_w = 0
        self._img_h = 0
        # 以原图像坐标为单位记录 top/bottom
        self._crop_top = 0
        self._crop_bottom = 0
        # 绘制目标区域（widget 坐标）
        self._target_rect = QRectF()
        # 拖拽相关
        self._dragging = None  # 'top' / 'bottom' / None
        self._drag_margin = 8  # 像素
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image_cv(self, frame_bgr):
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._image = qimg.copy()
        self._pixmap = QPixmap.fromImage(self._image)
        self._img_w, self._img_h = w, h
        # 默认上下各留 10px 或 5%
        margin = max(10, int(0.05 * h))
        self._crop_top = margin
        self._crop_bottom = h - margin
        self.update()

    def get_crop_bounds(self) -> Tuple[int, int]:
        return int(self._crop_top), int(self._crop_bottom)

    def _image_to_widget_y(self, y_img: float) -> float:
        if not self._target_rect.height():
            return 0
        scale = self._target_rect.height() / max(1, self._img_h)
        return self._target_rect.top() + y_img * scale

    def _widget_to_image_y(self, y_widget: float) -> float:
        if not self._target_rect.height():
            return 0
        scale = self._img_h / max(1e-6, self._target_rect.height())
        return (y_widget - self._target_rect.top()) * scale

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recalc_target_rect()

    def _recalc_target_rect(self):
        if not self._pixmap:
            self._target_rect = QRectF()
            return
        W = self.width()
        H = self.height()
        img_w, img_h = self._img_w, self._img_h
        if img_w <= 0 or img_h <= 0:
            self._target_rect = QRectF()
            return
        scale = min(W / img_w, H / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        left = (W - draw_w) / 2
        top = (H - draw_h) / 2
        self._target_rect = QRectF(left, top, draw_w, draw_h)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if not self._pixmap:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "未加载图像")
            return
        # 计算绘制区域
        self._recalc_target_rect()
        # 绘制图像
        painter.drawPixmap(self._target_rect, self._pixmap, QRectF(0, 0, self._img_w, self._img_h))

        # 计算两条线的 widget y
        y_top_w = self._image_to_widget_y(self._crop_top)
        y_bottom_w = self._image_to_widget_y(self._crop_bottom)

        # 阴影区域
        shade_color = QColor(0, 0, 0, 120)
        r = self.rect()
        painter.fillRect(QRectF(r.left(), r.top(), r.width(), y_top_w - r.top()), shade_color)
        painter.fillRect(QRectF(r.left(), y_bottom_w, r.width(), r.bottom() - y_bottom_w), shade_color)

        # 画线与抓手
        pen = QPen(QColor(0, 180, 255), 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(r.left(), y_top_w, r.right(), y_top_w)
        painter.drawLine(r.left(), y_bottom_w, r.right(), y_bottom_w)

        handle_pen = QPen(QColor(255, 255, 255), 6)
        painter.setPen(handle_pen)
        painter.drawPoint(QPointF(r.center().x(), y_top_w))
        painter.drawPoint(QPointF(r.center().x(), y_bottom_w))

    def mousePressEvent(self, event):
        if not self._pixmap:
            return
        y = event.pos().y()
        y_top_w = self._image_to_widget_y(self._crop_top)
        y_bottom_w = self._image_to_widget_y(self._crop_bottom)
        if abs(y - y_top_w) <= self._drag_margin:
            self._dragging = 'top'
        elif abs(y - y_bottom_w) <= self._drag_margin:
            self._dragging = 'bottom'
        else:
            self._dragging = None

    def mouseMoveEvent(self, event):
        if not self._pixmap or not self._dragging:
            return
        y_img = self._widget_to_image_y(event.pos().y())
        y_img = max(0, min(self._img_h - 1, y_img))
        if self._dragging == 'top':
            self._crop_top = min(y_img, self._crop_bottom - 1)
        elif self._dragging == 'bottom':
            self._crop_bottom = max(y_img, self._crop_top + 1)
        self.update()

    def mouseReleaseEvent(self, event):
        self._dragging = None


class CropDialog(QDialog):
    """弹窗：显示首帧并允许手动设置上下裁剪线。"""

    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动裁剪标记")
        self.resize(960, 540)

        self.selector = CropSelectorWidget(self)
        self.selector.set_image_cv(frame_bgr)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.selector, 1)
        layout.addWidget(btns)

    def get_bounds(self) -> Optional[Tuple[int, int]]:
        if self.result() == QDialog.Accepted:
            return self.selector.get_crop_bounds()
        return None


class Worker(QObject):
    finished = pyqtSignal(bool, str)
    message = pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str, manual_bounds: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.manual_bounds = manual_bounds

    def run(self):
        try:
            cropper = processing_mod.VideoFrameCropper()
            # 如果有手动 bounds，则覆盖其 process_frame
            if self.manual_bounds is not None:
                top, bottom = self.manual_bounds

                def _manual_pf(frame):
                    if frame is None:
                        return None
                    h, w = frame.shape[:2]
                    t = max(0, min(int(top), h - 2))
                    b = max(t + 1, min(int(bottom), h - 1))
                    cropper.crop_top = t
                    cropper.crop_bottom = b
                    cropper.original_height = h
                    cropper.original_width = w
                    # 不做自动检测
                    return None

                cropper.process_frame = _manual_pf  # 覆盖

            ok = cropper.process_video(self.input_path, self.output_path)
            self.finished.emit(bool(ok), self.output_path if ok else "")
        except Exception as e:
            self.message.emit(f"处理失败: {e}")
            self.finished.emit(False, "")


class VideoPlayerWidget(QWidget):
    """简单视频播放器：播放处理后的视频。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()

        # 控件
        self.play_btn = QPushButton()
        self.play_btn.setEnabled(False)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self._toggle_play)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)

        # 布局
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self.play_btn)
        ctrl_layout.addWidget(self.position_slider, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_widget, 1)
        layout.addLayout(ctrl_layout)

        self.player.setVideoOutput(self.video_widget)
        self.player.stateChanged.connect(self._update_state)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)

    def set_media(self, file_path: str):
        url = QUrl.fromLocalFile(os.path.abspath(file_path))
        self.player.setMedia(QMediaContent(url))
        self.play_btn.setEnabled(True)
        self.player.stop()
        self.position_slider.setValue(0)

    def _toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _update_state(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def _on_position(self, position):
        self.position_slider.setValue(position)

    def _on_duration(self, duration):
        self.position_slider.setRange(0, duration)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频裁剪处理 - UI")
        self.resize(1080, 720)

        # 输入输出
        self.input_edit = QLineEdit()
        self.input_browse = QPushButton("选择视频")
        self.input_browse.clicked.connect(self._choose_input)

        self.output_edit = QLineEdit()
        self.output_browse = QPushButton("选择输出")
        self.output_browse.clicked.connect(self._choose_output)

        # 操作按钮
        self.process_btn = QPushButton("自动裁剪处理")
        self.process_btn.clicked.connect(self._process_auto)
        self.manual_btn = QPushButton("不满意，手动标记")
        self.manual_btn.setEnabled(False)
        self.manual_btn.clicked.connect(self._open_manual_crop)

        self.reprocess_btn = QPushButton("使用手动裁剪重新处理")
        self.reprocess_btn.setEnabled(False)
        self.reprocess_btn.clicked.connect(self._process_manual)

        # 状态
        self.status_label = QLabel("就绪")

        # 播放器
        self.player = VideoPlayerWidget()

        # 顶部布局
        in_layout = QHBoxLayout()
        in_layout.addWidget(QLabel("输入:"))
        in_layout.addWidget(self.input_edit, 1)
        in_layout.addWidget(self.input_browse)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("输出:"))
        out_layout.addWidget(self.output_edit, 1)
        out_layout.addWidget(self.output_browse)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.process_btn)
        btn_layout.addWidget(self.manual_btn)
        btn_layout.addWidget(self.reprocess_btn)
        btn_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(in_layout)
        layout.addLayout(out_layout)
        layout.addLayout(btn_layout)
        layout.addWidget(self.player, 1)
        layout.addWidget(self.status_label)

        # 数据
        self._last_output: Optional[str] = None
        self._manual_bounds: Optional[Tuple[int, int]] = None
        self._first_frame_cache = None

    # 工具
    def _choose_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择输入视频", "", "视频文件 (*.mp4 *.mov *.avi *.mkv)")
        if not path:
            return
        self.input_edit.setText(path)
        # 自动填充输出路径
        base, ext = os.path.splitext(path)
        self.output_edit.setText(base + "_processed.mp4")
        self.status_label.setText("已选择输入视频")
        self.manual_btn.setEnabled(False)
        self.reprocess_btn.setEnabled(False)
        self._last_output = None
        self._manual_bounds = None
        self._first_frame_cache = None

    def _choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择输出位置", self.output_edit.text() or "output_video.mp4", "MP4 文件 (*.mp4)")
        if path:
            if not path.lower().endswith('.mp4'):
                path += '.mp4'
            self.output_edit.setText(path)

    def _validate_paths(self) -> Optional[Tuple[str, str]]:
        inp = self.input_edit.text().strip()
        outp = self.output_edit.text().strip()
        if not inp or not os.path.exists(inp):
            QMessageBox.warning(self, "提示", "请选择有效的输入视频文件")
            return None
        if not outp:
            QMessageBox.warning(self, "提示", "请设置输出路径")
            return None
        out_dir = os.path.dirname(outp)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "提示", f"无法创建输出目录: {e}")
                return None
        return inp, outp

    def _set_busy(self, busy: bool):
        self.process_btn.setEnabled(not busy)
        self.manual_btn.setEnabled(not busy and bool(self._last_output))
        self.reprocess_btn.setEnabled(not busy and self._manual_bounds is not None)
        self.input_browse.setEnabled(not busy)
        self.output_browse.setEnabled(not busy)
        self.status_label.setText("处理中..." if busy else "就绪")

    def _start_worker(self, input_path: str, output_path: str, manual_bounds: Optional[Tuple[int, int]] = None):
        self.thread = QThread(self)
        self.worker = Worker(input_path, output_path, manual_bounds)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished)
        self.worker.message.connect(self._on_message)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self._set_busy(True)
        self.thread.start()

    def _on_message(self, msg: str):
        self.status_label.setText(msg)

    def _on_finished(self, ok: bool, output_path: str):
        self._set_busy(False)
        if ok and output_path and os.path.exists(output_path):
            self._last_output = output_path
            self.player.set_media(output_path)
            self.status_label.setText("处理完成")
            self.manual_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "提示", "处理失败，请查看控制台输出或更换视频重试。")

    # 操作
    def _process_auto(self):
        v = self._validate_paths()
        if not v:
            return
        inp, outp = v
        self._start_worker(inp, outp, None)

    def _open_manual_crop(self):
        # 从输入视频读取首帧
        inp = self.input_edit.text().strip()
        if not inp or not os.path.exists(inp):
            QMessageBox.information(self, "提示", "请先选择有效的输入视频")
            return
        try:
            if self._first_frame_cache is None:
                cap = cv2.VideoCapture(inp)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    raise RuntimeError("无法读取首帧")
                self._first_frame_cache = frame
            dlg = CropDialog(self._first_frame_cache, self)
            if dlg.exec_() == QDialog.Accepted:
                bounds = dlg.get_bounds()
                if bounds is None:
                    return
                top, bottom = bounds
                if bottom - top < 2:
                    QMessageBox.information(self, "提示", "裁剪区域过小，请重新选择。")
                    return
                self._manual_bounds = (int(top), int(bottom))
                self.reprocess_btn.setEnabled(True)
                self.status_label.setText(f"已选择手动裁剪: top={top}, bottom={bottom}")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"读取首帧失败: {e}")

    def _process_manual(self):
        v = self._validate_paths()
        if not v:
            return
        if self._manual_bounds is None:
            QMessageBox.information(self, "提示", "请先手动选择裁剪范围")
            return
        inp, outp = v
        # 为避免覆盖自动结果，建议追加后缀
        base, ext = os.path.splitext(outp)
        outp2 = base + "_manual" + ext
        self.output_edit.setText(outp2)
        self._start_worker(inp, outp2, self._manual_bounds)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


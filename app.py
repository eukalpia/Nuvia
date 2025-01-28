import sys
import base64
import jax
import jax.numpy as jnp
import numpy as np
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QMessageBox)
from flax import linen as nn
from flax import serialization


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, 28, 28, 1)
        x = nn.Conv(32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)


class DrawingCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.setStyleSheet("background-color: black;")
        self._init_canvas()

    def _init_canvas(self):
        self._drawing = False
        self._last_pos = None
        self._image = QImage(400, 400, QImage.Format.Format_Grayscale8)
        self._image.fill(0)
        self._update_canvas()

    def mousePressEvent(self, event):
        self._drawing = True
        self._last_pos = event.position()
        self._draw_point(event.position())

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._draw_line(self._last_pos, event.position())
            self._last_pos = event.position()

    def mouseReleaseEvent(self, event):
        self._drawing = False
        self._last_pos = None

    def _draw_point(self, pos):
        painter = QPainter(self._image)
        painter.setPen(QPen(Qt.GlobalColor.white, 30, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPoint(int(pos.x()), int(pos.y()))
        self._update_canvas()

    def _draw_line(self, start, end):
        painter = QPainter(self._image)
        painter.setPen(QPen(Qt.GlobalColor.white, 30, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(int(start.x()), int(start.y()), int(end.x()), int(end.y()))
        self._update_canvas()

    def _update_canvas(self):
        self.setPixmap(QPixmap.fromImage(self._image))

    def clear(self):
        self._image.fill(0)
        self._update_canvas()

    def get_digit_data(self):
        scaled = self._image.scaled(28, 28, Qt.AspectRatioMode.IgnoreAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        buffer = scaled.bits().tobytes()
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape(28, 28)
        return arr.astype(np.float32) / 255.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание цифр")
        self.setFixedSize(500, 600)

        self.model = CNN()
        self.params = self._load_model()

        if self.params is not None:
            self._setup_ui()
            self._predict_fn = self._create_predict_fn()
            self._timer = QTimer()
            self._timer.timeout.connect(self._update_prediction)
            self._timer.start(100)
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить модель!")
            self.close()

    def _load_model(self):
        try:
            with open('model_base64.txt', 'r') as f:
                params_bytes = base64.b64decode(f.read())

            rng = jax.random.PRNGKey(0)
            init_params = self.model.init(rng, jnp.ones([1, 784]))['params']

            loaded_params = serialization.from_bytes(init_params, params_bytes)
            print("Модель успешно загружена")
            return loaded_params

        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки модели: {str(e)}")
            return None

    def _setup_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        self.pred_label = QLabel("Нарисуйте цифру")
        self.pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(self.pred_label)

        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.canvas.clear)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        layout.addWidget(clear_btn)

        central.setLayout(layout)
        central.setStyleSheet("background-color: #2b2b2b;")
        self.setCentralWidget(central)

    def _create_predict_fn(self):
        @jax.jit
        def predict(params, x):
            logits = self.model.apply({'params': params}, x)
            return jax.nn.softmax(logits)

        return predict

    def _update_prediction(self):
        try:
            digit_data = self.canvas.get_digit_data()
            x = jnp.array(digit_data.reshape(1, 784))
            probs = self._predict_fn(self.params, x)
            prediction = int(jnp.argmax(probs))
            confidence = float(probs[0, prediction])
            self.pred_label.setText(f"Цифра: {prediction} ({confidence:.1%})")
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
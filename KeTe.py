import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton
from keras.models import load_model
import numpy as np

class WebcamWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Load the model
        self.model = load_model(r"C:/Users/WSU/Desktop/new/converted_keras/keras_Model.h5", compile=False)

        # Load the labels
        with open(r"C:/Users/WSU/Desktop/new/converted_keras/labels.txt", "r") as f:
            self.class_names = f.readlines()

        # Set up the layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Set up the label to display the webcam image
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        # Set up the label to display prediction result
        self.prediction_label = QLabel()
        self.layout.addWidget(self.prediction_label)

        # Set up the button to close the application
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.close_button)

        # Start the webcam
        self.webcam = cv2.VideoCapture(0)

        # Start the timer to read frames from the webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # Update every 10 milliseconds

    def update_frame(self):
        # Read frame from webcam
        ret, frame = self.webcam.read()
        if not ret:
            return

        # Convert BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # Convert the frame to QImage
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update the image label
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        # Preprocess the frame for prediction
        frame_preprocessed = (frame_resized / 127.5) - 1
        frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)

        # Make prediction
        prediction = self.model.predict(frame_preprocessed)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        # Update prediction label
        prediction_text = f"Class: {class_name[2:]}, Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
        self.prediction_label.setText(prediction_text)

    def close_application(self):
        # Release webcam and stop the timer when closing the application
        self.webcam.release()
        self.timer.stop()
        sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WebcamWidget()
    window.setWindowTitle("Webcam Classifier")
    window.show()
    sys.exit(app.exec_())
